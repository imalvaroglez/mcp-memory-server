"""
Tests for TurboQuant compression engine.

Tests PolarQuant, QJL, and the combined TurboQuant pipeline
for correctness, compression ratio, and similarity preservation.
"""

import math
import numpy as np
import pytest

from mcp_memory.quantization import (
    TurboQuantCompressor,
    cartesian_to_polar,
    polar_to_cartesian,
    quantize_angles,
    dequantize_angles,
    qjl_compress,
    qjl_similarity_correction,
    turbo_compress,
    turbo_decompress,
    turbo_similarity,
    _get_projection_matrix,
)


# ---- PolarQuant Tests ----


class TestPolarQuant:
    """Tests for PolarQuant Cartesian-to-polar conversion and quantization."""

    def test_cartesian_to_polar_basic(self):
        """Test basic coordinate conversion."""
        vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        radii, angles = cartesian_to_polar(vec)

        assert len(radii) == 2
        assert len(angles) == 2
        # (1, 0) → radius=1, angle=0
        assert abs(radii[0] - 1.0) < 1e-5
        assert abs(angles[0] - 0.0) < 1e-5
        # (0, 1) → radius=1, angle=π/2
        assert abs(radii[1] - 1.0) < 1e-5
        assert abs(angles[1] - math.pi / 2) < 1e-5

    def test_polar_roundtrip_even_dimension(self):
        """Test that polar conversion round-trips for even-dim vectors."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        radii, angles = cartesian_to_polar(vec)
        reconstructed = polar_to_cartesian(radii, angles, 384)

        np.testing.assert_allclose(vec, reconstructed, atol=1e-5)

    def test_polar_roundtrip_odd_dimension(self):
        """Test that polar conversion handles odd-dimension vectors."""
        rng = np.random.RandomState(42)
        vec = rng.randn(385).astype(np.float32)

        radii, angles = cartesian_to_polar(vec)
        reconstructed = polar_to_cartesian(radii, angles, 385)

        np.testing.assert_allclose(vec, reconstructed, atol=1e-5)

    def test_quantize_dequantize_angles(self):
        """Test angle quantization round-trip accuracy."""
        angles = np.array([0.0, math.pi / 4, math.pi, 3 * math.pi / 2], dtype=np.float32)

        for bits in [2, 3, 4, 6, 8]:
            quantized = quantize_angles(angles, bits)
            dequantized = dequantize_angles(quantized, bits)

            # Dequantized should be close to original
            # (within half a quantization bin)
            bin_size = 2 * math.pi / (2**bits)
            for orig, deq in zip(angles, dequantized):
                error = abs(orig - deq)
                # Handle wrap-around at 2π
                error = min(error, 2 * math.pi - error)
                assert error <= bin_size, (
                    f"bits={bits}: angle {orig:.3f} → {deq:.3f}, error={error:.3f} "
                    f"> bin_size={bin_size:.3f}"
                )

    def test_quantize_preserves_range(self):
        """Test that quantized values stay in valid range."""
        angles = np.linspace(0, 2 * math.pi - 0.01, 100).astype(np.float32)
        for bits in [2, 4, 8]:
            quantized = quantize_angles(angles, bits)
            assert np.all(quantized >= 0)
            assert np.all(quantized < 2**bits)


# ---- QJL Tests ----


class TestQJL:
    """Tests for Quantized Johnson-Lindenstrauss compression."""

    def test_qjl_compress_produces_packed_bits(self):
        """Test that QJL produces correctly packed bit arrays."""
        rng = np.random.RandomState(42)
        residual = rng.randn(384).astype(np.float32)
        proj = _get_projection_matrix(384, n_projections=128)

        packed = qjl_compress(residual, proj)

        # 128 bits → 16 bytes
        assert len(packed) == 16

    def test_qjl_similarity_identical(self):
        """Test that QJL reports high similarity for identical vectors."""
        rng = np.random.RandomState(42)
        residual = rng.randn(384).astype(np.float32)
        proj = _get_projection_matrix(384, n_projections=128)

        packed = qjl_compress(residual, proj)
        sim = qjl_similarity_correction(packed, packed, n_projections=128)

        # Identical vectors should have high similarity
        assert sim > 0.8

    def test_qjl_similarity_orthogonal(self):
        """Test that QJL reports low similarity for orthogonal vectors."""
        rng = np.random.RandomState(42)
        v1 = np.zeros(384, dtype=np.float32)
        v2 = np.zeros(384, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0

        proj = _get_projection_matrix(384, n_projections=256)

        packed1 = qjl_compress(v1, proj)
        packed2 = qjl_compress(v2, proj)

        sim = qjl_similarity_correction(packed1, packed2, n_projections=256)
        # Orthogonal vectors should have near-zero similarity
        assert abs(sim) < 0.4


# ---- TurboQuant Pipeline Tests ----


class TestTurboQuant:
    """Tests for the combined TurboQuant compression pipeline."""

    def test_compress_decompress_preserves_shape(self):
        """Test that compression/decompression preserves vector dimension."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        compressor = TurboQuantCompressor(dimension=384, bits=4)
        compressed = compressor.compress(vec)
        decompressed = compressor.decompress(compressed)

        assert len(decompressed) == 384

    def test_compress_achieves_size_reduction(self):
        """Test that compressed representation is smaller than float32."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        compressor = TurboQuantCompressor(dimension=384, bits=4)
        compressed = compressor.compress(vec)

        original_size = 384 * 4  # float32 = 4 bytes each
        compressed_size = len(compressed)

        assert compressed_size < original_size
        ratio = original_size / compressed_size
        # Should achieve at least 1.5x compression
        assert ratio > 1.5, f"Compression ratio {ratio:.2f}x is too low"

    def test_similarity_high_for_identical_vectors(self):
        """Test similarity is near 1.0 for identical vectors."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        compressor = TurboQuantCompressor(dimension=384, bits=4)
        compressed = compressor.compress(vec)

        sim = compressor.similarity(compressed, compressed)
        assert sim > 0.95, f"Self-similarity {sim:.3f} is too low"

    def test_similarity_preserves_order(self):
        """Test that TurboQuant similarity preserves relative ordering."""
        rng = np.random.RandomState(42)
        base = rng.randn(384).astype(np.float32)
        similar = base + rng.randn(384).astype(np.float32) * 0.1  # Small perturbation
        different = rng.randn(384).astype(np.float32)  # Completely different

        compressor = TurboQuantCompressor(dimension=384, bits=4)

        comp_base = compressor.compress(base)
        comp_similar = compressor.compress(similar)
        comp_different = compressor.compress(different)

        sim_close = compressor.similarity(comp_base, comp_similar)
        sim_far = compressor.similarity(comp_base, comp_different)

        assert sim_close > sim_far, (
            f"Similar vector sim ({sim_close:.3f}) should be > "
            f"different vector sim ({sim_far:.3f})"
        )

    def test_similarity_correlates_with_exact(self):
        """Test that compressed similarity correlates with exact cosine similarity."""
        rng = np.random.RandomState(42)
        compressor = TurboQuantCompressor(dimension=384, bits=4)

        exact_sims = []
        turbo_sims = []

        for _ in range(50):
            a = rng.randn(384).astype(np.float32)
            b = rng.randn(384).astype(np.float32)

            # Exact cosine similarity
            exact = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            exact_sims.append(exact)

            # TurboQuant similarity
            comp_a = compressor.compress(a)
            comp_b = compressor.compress(b)
            turbo = compressor.similarity(comp_a, comp_b)
            turbo_sims.append(turbo)

        # Correlation between exact and compressed similarities
        correlation = np.corrcoef(exact_sims, turbo_sims)[0, 1]
        assert correlation > 0.95, (
            f"Similarity correlation {correlation:.3f} is below 0.95 threshold"
        )

    def test_batch_compression(self):
        """Test batch compression produces correct number of outputs."""
        rng = np.random.RandomState(42)
        vectors = rng.randn(20, 384).astype(np.float32)

        compressor = TurboQuantCompressor(dimension=384, bits=4)
        compressed = compressor.compress_batch(vectors)

        assert len(compressed) == 20
        assert all(isinstance(c, bytes) for c in compressed)

    def test_compression_stats(self):
        """Test that compression stats are computed correctly."""
        rng = np.random.RandomState(42)
        vectors = rng.randn(10, 384).astype(np.float32)

        compressor = TurboQuantCompressor(dimension=384, bits=4)
        stats = compressor.compression_stats(vectors)

        assert stats.total_vectors == 10
        assert stats.original_bytes == 10 * 384 * 4
        assert stats.compressed_bytes > 0
        assert stats.compression_ratio > 1.0
        assert stats.avg_compression_ms > 0
        assert 0.9 < stats.similarity_preservation <= 1.0

    def test_different_bit_widths(self):
        """Test compression with different bit widths."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        for bits in [2, 3, 4, 6, 8]:
            compressor = TurboQuantCompressor(dimension=384, bits=bits)
            compressed = compressor.compress(vec)
            decompressed = compressor.decompress(compressed)

            # Check similarity preservation improves with more bits
            sim = float(
                np.dot(vec, decompressed)
                / (np.linalg.norm(vec) * np.linalg.norm(decompressed))
            )
            assert sim > 0.8, f"bits={bits}: similarity {sim:.3f} is too low"


# ---- Convenience Function Tests ----


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_turbo_compress_decompress(self):
        """Test the turbo_compress/turbo_decompress convenience functions."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)

        compressed = turbo_compress(vec)
        decompressed = turbo_decompress(compressed)

        assert len(decompressed) == 384

        sim = float(
            np.dot(vec, decompressed)
            / (np.linalg.norm(vec) * np.linalg.norm(decompressed))
        )
        assert sim > 0.95

    def test_turbo_similarity(self):
        """Test the turbo_similarity convenience function."""
        rng = np.random.RandomState(42)
        a = rng.randn(384).astype(np.float32)
        b = rng.randn(384).astype(np.float32)

        comp_a = turbo_compress(a)
        comp_b = turbo_compress(b)

        sim = turbo_similarity(comp_a, comp_b)
        assert -1.0 <= sim <= 1.0
