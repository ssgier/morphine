use crate::{params::StpParams, util::get_decay_factor};

pub trait ShortTermPlasticity {
    fn on_pre_syn_spike_get_value(&mut self, t: usize) -> f32;
}

pub fn create(std_params: &StpParams) -> Box<dyn ShortTermPlasticity + Send> {
    match *std_params {
        StpParams::NoStp => Box::new(Constant::new()),
        StpParams::Depression { tau, p0, factor } => Box::new(Depression {
            tau,
            p0,
            factor,
            last_t: 0,
            last_value: p0,
        }),
        StpParams::Facilitation { tau, p0, factor } => Box::new(Facilitation {
            tau,
            p0,
            factor,
            last_t: 0,
            last_value: p0,
        }),
    }
}

struct Constant;

struct Depression {
    p0: f32,
    factor: f32,
    tau: f32,
    last_t: usize,
    last_value: f32,
}

struct Facilitation {
    p0: f32,
    factor: f32,
    tau: f32,
    last_t: usize,
    last_value: f32,
}

impl Constant {
    pub fn new() -> Self {
        Self
    }
}

impl ShortTermPlasticity for Constant {
    fn on_pre_syn_spike_get_value(&mut self, _t: usize) -> f32 {
        1.0
    }
}

impl ShortTermPlasticity for Depression {
    fn on_pre_syn_spike_get_value(&mut self, t: usize) -> f32 {
        let decay_factor = get_decay_factor(t, self.last_t, self.tau);
        let value = decay_factor * self.last_value + (1.0 - decay_factor) * self.p0;
        let value_after_transmission = value - self.factor * value;
        self.last_value = value_after_transmission;
        self.last_t = t;
        value
    }
}

impl ShortTermPlasticity for Facilitation {
    fn on_pre_syn_spike_get_value(&mut self, t: usize) -> f32 {
        let decay_factor = get_decay_factor(t, self.last_t, self.tau);
        let value = decay_factor * self.last_value + (1.0 - decay_factor) * self.p0;
        let value_after_transmission = value + self.factor * (1.0 - value);
        self.last_value = value_after_transmission;
        self.last_t = t;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn no_stp() {
        let mut stp = create(&StpParams::NoStp);
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(4), 1.0);
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(10), 1.0);
    }

    #[test]
    fn depression() {
        let params = StpParams::Depression {
            tau: 10.0,
            p0: 0.4,
            factor: 0.6,
        };

        let mut stp = create(&params);
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(4), 0.4);

        let depressed_value = stp.on_pre_syn_spike_get_value(9);
        assert_approx_eq!(f32, depressed_value, 0.4 - 0.4 * 0.6 * (-0.5f32).exp());

        let more_depressed_value = stp.on_pre_syn_spike_get_value(9);
        assert_approx_eq!(f32, more_depressed_value, depressed_value * (1.0 - 0.6));

        let decayed_close_to_resting_state = stp.on_pre_syn_spike_get_value(100);
        let most_depressed_value = more_depressed_value * (1.0 - 0.6);
        let offset = most_depressed_value - 0.4;
        assert_approx_eq!(
            f32,
            decayed_close_to_resting_state,
            0.4 + offset * (-9.1f32).exp()
        );

        // no efficacy
        for _ in 0..100 {
            stp.on_pre_syn_spike_get_value(100);
        }
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(100), 0.0);

        // back to resting state again
        assert_approx_eq!(
            f32,
            stp.on_pre_syn_spike_get_value(200),
            0.4 - 0.4 * (-10.0f32).exp()
        );
    }

    #[test]
    fn facilitation() {
        let params = StpParams::Facilitation {
            tau: 10.0,
            p0: 0.4,
            factor: 0.6,
        };

        let mut stp = create(&params);
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(4), 0.4);

        let facilitated_value = stp.on_pre_syn_spike_get_value(9);
        assert_approx_eq!(
            f32,
            facilitated_value,
            0.4 + (1.0 - 0.4) * 0.6 * (-0.5f32).exp()
        );

        let more_facilitated_value = stp.on_pre_syn_spike_get_value(9);
        assert_approx_eq!(
            f32,
            more_facilitated_value,
            facilitated_value + (1.0 - facilitated_value) * 0.6
        );

        let decayed_close_to_resting_state = stp.on_pre_syn_spike_get_value(100);
        let most_facilitated_value = more_facilitated_value + (1.0 - more_facilitated_value) * 0.6;
        let offset = most_facilitated_value - 0.4;
        assert_approx_eq!(
            f32,
            decayed_close_to_resting_state,
            0.4 + offset * (-9.1f32).exp()
        );

        // full efficacy
        for _ in 0..100 {
            stp.on_pre_syn_spike_get_value(100);
        }
        assert_approx_eq!(f32, stp.on_pre_syn_spike_get_value(100), 1.0);

        // back to resting state again
        assert_approx_eq!(
            f32,
            stp.on_pre_syn_spike_get_value(200),
            0.4 + 0.6 * (-10.0f32).exp()
        );
    }
}
