import sys
sys.path.append('..')

import unittest

import random, numpy as np

from mixins import SeedableMixin

class SeedableDerived(SeedableMixin):
    def __init__(self):
        self.foo = 'foo'
        # Doesn't call super().__init__()! Should still work in this case.

    def gen_random_num(self):
        # This is purely random
        return (random.random(), np.random.rand())

    @SeedableMixin.WithSeed(key="decorated")
    def decorated_gen_random_num(self):
        return (random.random(), np.random.rand())

    @SeedableMixin.WithSeed
    def decorated_auto_key(self):
        return random.random()

class TestSeedableMixin(unittest.TestCase):
    def test_constructs(self):
        T = SeedableMixin()
        T = SeedableDerived()

    def test_responds_to_methods(self):
        T = SeedableMixin()

        T._seed()
        T._last_seed('foo')

        T = SeedableDerived()
        T._seed()
        T._last_seed('foo')


    def test_seeding_freezes_randomness(self):
        T = SeedableDerived()

        unseeded_1 = T.gen_random_num()
        unseeded_2 = T.gen_random_num()

        # Without seeding, repeated calls should be different.
        self.assertNotEqual(unseeded_1, unseeded_2)

        T._seed(1)
        seeded_1_1 = T.gen_random_num()
        seeded_2_1 = T.gen_random_num()

        # Even if I seeded at the start, repeated calls should still be different.
        self.assertNotEqual(seeded_1_1, seeded_2_1)

        T._seed(1)
        seeded_1_2 = T.gen_random_num()
        seeded_2_2 = T.gen_random_num()

        # Since I seeded again, they should match the prior sequence.
        self.assertEqual(seeded_1_1, seeded_1_2)
        self.assertEqual(seeded_2_1, seeded_2_2)

    def test_decorated_seeding_freezes_randomness(self):
        T = SeedableDerived()

        unseeded_1 = T.decorated_gen_random_num()
        unseeded_2 = T.decorated_gen_random_num()

        # Without seeding, repeated calls should be different.
        self.assertNotEqual(unseeded_1, unseeded_2)

        seeded_1_1 = T.decorated_gen_random_num(seed=1)
        seeded_2_1 = T.decorated_gen_random_num(seed=2)

        # Even if I seeded at the start, repeated calls should still be different.
        self.assertNotEqual(seeded_1_1, seeded_2_1)

        seeded_1_2 = T.decorated_gen_random_num(seed=1)
        seeded_2_2 = T.decorated_gen_random_num(seed=2)

        # Since they are seeded, they should match the prior sequence.
        self.assertEqual(seeded_1_1, seeded_1_2)
        self.assertEqual(seeded_2_1, seeded_2_2)

        # Now we want to make sure the seeding is consistent even interrupted.

        T._seed(0)
        seeded_1_3 = T.decorated_gen_random_num(seed=1)
        T._seed(10)
        seeded_2_3 = T.decorated_gen_random_num(seed=2)

        self.assertEqual(seeded_1_1, seeded_1_3)
        self.assertEqual(seeded_2_1, seeded_2_3)


    def test_seeds_follow_consistent_sequence(self):
        T = SeedableDerived()

        unseeded_seq = [T._seed() for i in range(5)]

        seed_1 = T._seed(1)

        # seed_1 should be 1 given I passed a seed in:
        self.assertEqual(seed_1, 1)

        next_seeds_1 = [T._seed() for i in range(5)]

        # These should differ from the unseeded sequence of seeds
        self.assertNotEqual(unseeded_seq, next_seeds_1)

        T._seed(1)

        next_seeds_2 = [T._seed() for i in range(5)]

        # The sequence of seeds should be the same here.
        self.assertEqual(next_seeds_1, next_seeds_2)

    def test_get_last_seed(self):
        T = SeedableDerived()

        key = 'key'
        non_key = 'not_key'

        seed_key_early = 1
        seed_key_late  = 1
        seed_non_key   = 2

        T._seed()

        idx, seed = T._last_seed(key)
        self.assertEqual(idx, -1)
        self.assertEqual(seed, None)

        T._seed(seed_key_early, key)
        T._seed()
        T._seed(seed_non_key, non_key)
        T._seed(seed_key_late, key)
        T._seed(seed_non_key, non_key)

        idx, seed = T._last_seed(key)
        self.assertEqual(idx, 4)
        self.assertEqual(seed, seed_key_late)

if __name__ == '__main__': unittest.main()

