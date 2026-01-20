#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdbool.h>

#include <atomic_bitset.h>


static inline AtomicBitset* bitset_create(size_t nbits) {
    AtomicBitset* bs = malloc(sizeof(*bs));
    if (!bs) return NULL;

    bs->nbits = nbits;
    bs->nwords = (nbits + 63) / 64;

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

static inline void bitset_free(AtomicBitset* bs) {
    if (!bs) {
        return;
    }

    free(bs->words);
    free(bs);
}

static inline void bitset_set(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;

    atomic_fetch_or(&bs->words[WORD_INDEX(i)], BIT_MASK(i));
}

static inline void bitset_clear(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;

    atomic_fetch_and(&bs->words[WORD_INDEX(i)], ~BIT_MASK(i));
}

static inline void bitset_set_relaxed(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;

    atomic_fetch_or_explicit(&bs->words[WORD_INDEX(i)], BIT_MASK(i), memory_order_relaxed);
}

static inline void bitset_clear_relaxed(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;

    atomic_fetch_and_explicit(&bs->words[WORD_INDEX(i)], ~BIT_MASK(i), memory_order_relaxed);
}

static inline bool bitset_test(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return false;

    return (atomic_load(&bs->words[WORD_INDEX(i)]) & BIT_MASK(i)) != 0;
}

static inline void bitset_clear_all(AtomicBitset *bs) {
    for (size_t i = 0; i < bs->nwords; i++) {
        atomic_store(&bs->words[i], 0);
    }
}
