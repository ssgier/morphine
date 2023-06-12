pub struct DistanceCalculator {
    grid_size_from: usize,
    grid_size_to: usize,
    grid_delta_from: f64,
    grid_delta_to: f64,
    dim: usize,
    hyper_sphere: bool,
    grid_shift: f64,
}

impl DistanceCalculator {
    fn calculate_grid_size(dim: usize, num_cells: usize) -> usize {
        (num_cells as f64).powf(1.0 / dim as f64).round() as usize
    }

    fn calculate_grid_delta(grid_size: usize, hyper_sphere: bool) -> f64 {
        if hyper_sphere {
            if grid_size >= 1 {
                1.0 / grid_size as f64
            } else {
                f64::NAN
            }
        } else if grid_size >= 2 {
            1.0 / (grid_size - 1) as f64
        } else {
            f64::NAN
        }
    }

    fn calculate_position_component(
        component_idx: usize,
        grid_size: usize,
        grid_delta: f64,
        grid_shift: f64,
    ) -> f64 {
        if grid_size == 1 {
            0.5
        } else {
            (component_idx as f64 + grid_shift) * grid_delta
        }
    }

    pub fn new(dim: usize, hyper_sphere: bool, num_cells_from: usize, num_cells_to: usize) -> Self {
        let grid_size_from = Self::calculate_grid_size(dim, num_cells_from);
        let grid_size_to = Self::calculate_grid_size(dim, num_cells_to);
        let grid_shift = if hyper_sphere { 0.5 } else { 0.0 };
        Self {
            grid_size_from,
            grid_size_to,
            grid_delta_from: Self::calculate_grid_delta(grid_size_from, hyper_sphere),
            grid_delta_to: Self::calculate_grid_delta(grid_size_to, hyper_sphere),
            dim,
            hyper_sphere,
            grid_shift,
        }
    }

    pub fn calculate_distance(&self, from_idx: usize, to_idx: usize) -> f64 {
        let mut from_remainder = from_idx;
        let mut to_remainder = to_idx;

        let mut squared_distance = 0.0;

        for _ in 0..self.dim {
            let from_component_idx = from_remainder % self.grid_size_from;
            let to_component_idx = to_remainder % self.grid_size_to;

            let from_component = Self::calculate_position_component(
                from_component_idx,
                self.grid_size_from,
                self.grid_delta_from,
                self.grid_shift,
            );

            let to_component = Self::calculate_position_component(
                to_component_idx,
                self.grid_size_to,
                self.grid_delta_to,
                self.grid_shift,
            );

            let min_component = from_component.min(to_component);
            let max_component = from_component.max(to_component);

            let mut distance_component = max_component - min_component;

            if self.hyper_sphere {
                distance_component = distance_component.min(min_component + 1.0 - max_component);
            }

            squared_distance += distance_component * distance_component;

            from_remainder -= from_component_idx;
            to_remainder -= to_component_idx;

            from_remainder /= self.grid_size_from;
            to_remainder /= self.grid_size_to;
        }
        
        squared_distance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::DistanceCalculator;
    use float_cmp::assert_approx_eq;

    #[test]
    fn symmetry() {
        for dim in 0..=4usize {
            for i in 1..4usize {
                let grid_size = i.pow(dim as u32);

                let sut_normal = DistanceCalculator::new(dim, false, grid_size, grid_size);
                let sut_hyper_sphere = DistanceCalculator::new(dim, true, grid_size, grid_size);

                for from_idx in 0..grid_size {
                    for to_idx in 0..grid_size {
                        assert_approx_eq!(
                            f64,
                            sut_normal.calculate_distance(from_idx, to_idx),
                            sut_normal.calculate_distance(to_idx, from_idx)
                        );
                        assert_approx_eq!(
                            f64,
                            sut_hyper_sphere.calculate_distance(from_idx, to_idx),
                            sut_hyper_sphere.calculate_distance(to_idx, from_idx)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dim_0() {
        let sut_normal = DistanceCalculator::new(0, false, 2, 3);
        let sut_hyper_sphere = DistanceCalculator::new(0, true, 2, 3);

        for from_idx in 0..2 {
            for to_idx in 0..3 {
                assert_approx_eq!(f64, sut_normal.calculate_distance(from_idx, to_idx), 0.0);
                assert_approx_eq!(
                    f64,
                    sut_hyper_sphere.calculate_distance(from_idx, to_idx),
                    0.0
                );
            }
        }
    }

    #[test]
    fn dim_1() {
        let sut = DistanceCalculator::new(1, false, 2, 1);
        assert_approx_eq!(f64, sut.calculate_distance(0, 0), 0.5);
        assert_approx_eq!(f64, sut.calculate_distance(1, 0), 0.5);

        let sut = DistanceCalculator::new(1, true, 2, 1);
        assert_approx_eq!(f64, sut.calculate_distance(0, 0), 0.25);
        assert_approx_eq!(f64, sut.calculate_distance(1, 0), 0.25);

        let sut = DistanceCalculator::new(1, true, 2, 2);
        assert_approx_eq!(f64, sut.calculate_distance(0, 0), 0.0);
        assert_approx_eq!(f64, sut.calculate_distance(0, 1), 0.5);

        let sut = DistanceCalculator::new(1, true, 2, 3);
        assert_approx_eq!(f64, sut.calculate_distance(0, 0), 0.25 - 1.0 / 6.0);
        assert_approx_eq!(f64, sut.calculate_distance(0, 1), 0.25);

        let sut = DistanceCalculator::new(1, false, 5, 6);
        assert_approx_eq!(f64, sut.calculate_distance(1, 1), 0.25 - 0.2);
        assert_approx_eq!(f64, sut.calculate_distance(1, 4), 0.8 - 0.25);
        assert_approx_eq!(f64, sut.calculate_distance(1, 5), 1.0 - 0.25);

        let sut = DistanceCalculator::new(1, true, 5, 6);
        assert_approx_eq!(f64, sut.calculate_distance(1, 1), 0.3 - 1.5 / 6.0);
        assert_approx_eq!(f64, sut.calculate_distance(1, 5), 0.3 + 0.5 / 6.0);
    }

    #[test]
    fn dim_2() {
        let sut = DistanceCalculator::new(2, true, 4, 1);
        for from_idx in 0..4 {
            assert_approx_eq!(f64, sut.calculate_distance(from_idx, 0), 2f64.sqrt() * 0.25);
        }

        let sut = DistanceCalculator::new(2, false, 121, 100);
        let diff_dim_0 = 0.9 - 1.0 / 9.0;
        let diff_dim_1: f64 = 0.8 - 2.0 / 9.0;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(19, 82),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        );

        let sut = DistanceCalculator::new(2, true, 100, 81);
        let diff_dim_0 = 0.05 + 0.5 / 9.0;
        let diff_dim_1: f64 = 0.5 / 9.0 - 0.05;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(9, 0),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        );

        let diff_dim_0 = 0.05 + 1.5 / 9.0;
        let diff_dim_1: f64 = 0.5 / 9.0 - 0.05;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(9, 1),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        );

        let diff_dim_0 = 2.5 / 9.0 - 0.15;
        let diff_dim_1: f64 = 1.5 / 9.0 - 0.15;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(11, 11),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        );

        let diff_dim_0 = 0.15 + 1.5 / 9.0;
        let diff_dim_1: f64 = diff_dim_0;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(18, 64),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        );

        let diff_dim_0 = 6.5 / 9.0 - 0.45;
        let diff_dim_1: f64 = 6.5 / 9.0 - 0.35;
        assert_approx_eq!(
            f64,
            sut.calculate_distance(34, 60),
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1).sqrt()
        )
    }

    #[test]
    fn dim_3() {
        let num_cells = 1000;
        let from_idx = 9 * 100 + 9 * 10 + 2;
        let to_idx = 6 * 100 + 2 * 10 + 4;

        let sut = DistanceCalculator::new(3, false, num_cells, num_cells);
        let diff_dim_0 = 3.0 / 9.0;
        let diff_dim_1 = 7.0 / 9.0;
        let diff_dim_2 = 2f64 / 9.0;

        let expected_distance =
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1 + diff_dim_2 * diff_dim_2).sqrt();

        assert_approx_eq!(
            f64,
            sut.calculate_distance(from_idx, to_idx),
            expected_distance
        );

        let sut = DistanceCalculator::new(3, true, num_cells, num_cells);
        let diff_dim_0 = 0.3;
        let diff_dim_1 = 0.3;
        let diff_dim_2 = 0.2f64;

        let expected_distance =
            (diff_dim_0 * diff_dim_0 + diff_dim_1 * diff_dim_1 + diff_dim_2 * diff_dim_2).sqrt();

        assert_approx_eq!(
            f64,
            sut.calculate_distance(from_idx, to_idx),
            expected_distance
        );
    }
}
