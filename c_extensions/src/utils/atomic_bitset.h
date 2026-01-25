#ifndef ATOMIC_BITSET_H
#define ATOMIC_BITSET_H

#define PY_SSIZE_T_CLEAN
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdbool.h>


typedef struct {
    size_t nbits;
    size_t nwords;
    _Atomic(uint64_t)* words;
} AtomicBitset;

#define WORD_INDEX(i) ((i) >> 6) // i / 64
#define BIT_MASK(i)  (1ULL << (i % 64)) // (i % 64)th bit

AtomicBitset* bitset_create(size_t nbits);

static inline void bitset_free(AtomicBitset* bs) {
    if (!bs) return;
    free(bs->words);
    free(bs);
}

static inline void bitset_clear(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;
    atomic_fetch_and(&bs->words[WORD_INDEX(i)], ~BIT_MASK(i));
}

static inline void bitset_clear_relaxed(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;
    atomic_fetch_and_explicit(&bs->words[WORD_INDEX(i)], ~BIT_MASK(i), memory_order_relaxed);
}

static inline void bitset_set_relaxed(AtomicBitset* bs, size_t i) {
    if (i >= bs->nbits) return;
    atomic_fetch_or_explicit(&bs->words[WORD_INDEX(i)], BIT_MASK(i), memory_order_relaxed);
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

#endif 