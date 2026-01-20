#ifndef ATOMIC_BITSET_H
#define ATOMIC_BITSET_H

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
void bitset_free(AtomicBitset* bs);

void bitset_set(AtomicBitset* bs, size_t i);
void bitset_clear(AtomicBitset* bs, size_t i);
void bitset_set_relaxed(AtomicBitset* bs, size_t i);
void bitset_clear_relaxed(AtomicBitset* bs, size_t i);
bool bitset_test(AtomicBitset* bs, size_t i);
void bitset_clear_all(AtomicBitset* bs);

#endif 