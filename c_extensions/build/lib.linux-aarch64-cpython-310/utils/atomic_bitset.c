#include "atomic_bitset.h"


AtomicBitset* bitset_create(size_t nbits) {
    AtomicBitset* bs = malloc(sizeof(*bs));
    if (!bs) return NULL;

    bs->nbits = nbits;
    bs->nwords = (nbits + 63) / 64;

    // Align memory for cache efficiency
    size_t bytes = ((bs->nwords * sizeof(_Atomic(uint64_t)) + 63) / 64) * 64;
    bs->words = aligned_alloc(64, bytes);
    if (!bs->words) {
        free(bs);
        return NULL;
    }

    for (size_t i = 0; i < bs->nwords; i++) {
        atomic_init(&bs->words[i], 0);
    }

    return bs;
}