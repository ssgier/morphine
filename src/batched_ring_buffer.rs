#[derive(Debug)]
pub struct BatchedRingBuffer<T> {
    batch_buffers: Vec<Vec<T>>,
    current_pos: usize,
}

impl<T: Clone> BatchedRingBuffer<T> {
    pub fn new(buffer_size: usize) -> BatchedRingBuffer<T> {
        BatchedRingBuffer {
            batch_buffers: vec![Vec::new(); buffer_size],
            current_pos: 0,
        }
    }

    #[cfg(debug_assertions)]
    fn is_within_horizon(&self, offset: usize) -> bool {
        offset < self.batch_buffers.len()
    }

    pub fn push_at_offset(&mut self, offset: usize, value: T) {
        #[cfg(debug_assertions)]
        assert!(self.is_within_horizon(offset));

        let target_pos = self.get_target_pos(offset);
        self.batch_buffers[target_pos].push(value);
    }

    fn get_target_pos(&self, offset: usize) -> usize {
        let mut target_pos = self.current_pos + offset;

        if target_pos >= self.batch_buffers.len() {
            target_pos -= self.batch_buffers.len();
        }

        target_pos
    }

    pub fn next_batch_size(&self) -> usize {
        self.batch_buffers[self.current_pos].len()
    }

    pub fn drain_and_advance(&mut self) -> impl Iterator<Item = T> + '_ {
        let pos = self.current_pos;
        self.current_pos += 1;
        if self.current_pos == self.batch_buffers.len() {
            self.current_pos -= self.batch_buffers.len();
        }
        self.batch_buffers[pos].drain(..)
    }
}

#[cfg(test)]
mod tests {

    use super::BatchedRingBuffer;
    use itertools::assert_equal;
    use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

    const EMPTY: [i32; 0] = [];

    #[test]
    fn empty() {
        let mut sut: BatchedRingBuffer<i32> = BatchedRingBuffer::new(8);
        assert_equal(sut.drain_and_advance(), EMPTY);
    }

    #[test]
    fn single_element() {
        let mut sut: BatchedRingBuffer<i32> = BatchedRingBuffer::new(8);
        sut.push_at_offset(1, 11);
        assert_eq!(sut.next_batch_size(), 0);
        assert_equal(sut.drain_and_advance(), EMPTY);
        assert_eq!(sut.next_batch_size(), 1);
        assert_equal(sut.drain_and_advance(), [11]);
        assert_equal(sut.drain_and_advance(), EMPTY);
    }

    #[test]
    fn round_trip() {
        let mut sut: BatchedRingBuffer<i32> = BatchedRingBuffer::new(10);
        let first_pass_value = 2;
        let second_pass_value = 3;
        sut.push_at_offset(1, first_pass_value);

        #[allow(unused_must_use)]
        for _ in 0..2 {
            sut.drain_and_advance();
        }

        sut.push_at_offset(9, second_pass_value);
        for _ in 0..9 {
            assert_equal(sut.drain_and_advance(), EMPTY);
        }
        assert_equal(sut.drain_and_advance(), [second_pass_value]);
    }

    #[test]
    fn randomized_input() {
        let mut sut: BatchedRingBuffer<i32> = BatchedRingBuffer::new(10);
        const NUM_TIME_SLOTS: usize = 101;
        let mut flat_expected_data = vec![Vec::<i32>::new(); NUM_TIME_SLOTS];
        let mut rng = StdRng::seed_from_u64(0);
        let amount_dist = Uniform::from(0..10);
        let offset_dist = Uniform::from(1..10);
        let value_dist = Uniform::from(-1000..1000);

        for flat_loc in 0..NUM_TIME_SLOTS {
            let amount = amount_dist.sample(&mut rng);

            for _ in 0..amount {
                let offset = offset_dist.sample(&mut rng);
                let value = value_dist.sample(&mut rng);
                sut.push_at_offset(offset, value);
                let target_loc_flat_data = flat_loc + offset;
                if target_loc_flat_data < NUM_TIME_SLOTS {
                    flat_expected_data[target_loc_flat_data].push(value);
                }
            }

            assert_eq!(sut.next_batch_size(), flat_expected_data[flat_loc].len());

            assert!(sut
                .drain_and_advance()
                .eq(flat_expected_data[flat_loc].drain(..)));
        }
    }
}
