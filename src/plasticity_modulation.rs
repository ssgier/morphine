use std::{collections::VecDeque, usize};

use crate::{params::PlasticityModulationParams, util};

#[derive(Debug)]
pub struct PlasticityModulator {
    elig_traces: VecDeque<EligibilityTrace>,
    dopamine_buffer: Vec<DopamineBufferItem>,
    pending_dopamine_buffer_item: DopamineBufferItem,
    params: PlasticityModulationParams,
    next_dopamine_buffer_flush_t: usize,
}

#[derive(Debug, PartialEq)]
pub struct PlasticityEvent {
    pub weight_change: f32,
    pub pre_syn_nid: usize,
    pub synapse_idx: usize,
}

pub struct PlasticityEvents<'a> {
    dopamine_buffer: Vec<DopamineBufferItem>,
    elig_traces: &'a VecDeque<EligibilityTrace>,
    tau_eligibility_trace: f32,
}

impl<'a> PlasticityEvents<'a> {
    fn new(
        inner: &'a VecDeque<EligibilityTrace>,
        dopamine_buffer: Vec<DopamineBufferItem>,
        tau_eligibility_trace: f32,
    ) -> PlasticityEvents<'a> {
        Self {
            dopamine_buffer,
            elig_traces: inner,
            tau_eligibility_trace,
        }
    }

    fn get_weight_change_contrib(
        &self,
        dopamine_buffer_item: &DopamineBufferItem,
        elig_trace: &EligibilityTrace,
    ) -> f32 {
        if dopamine_buffer_item.t_release >= elig_trace.t {
            let decay_factor = util::get_decay_factor(
                dopamine_buffer_item.t_release,
                elig_trace.t,
                self.tau_eligibility_trace,
            );

            elig_trace.stdp_value * decay_factor * dopamine_buffer_item.scaled_amount
        } else {
            0.0
        }
    }

    pub fn iter(&'a self) -> impl Iterator<Item = PlasticityEvent> + 'a {
        self.elig_traces
            .iter()
            .map(|elig_trace| {
                let weight_change = self
                    .dopamine_buffer
                    .iter()
                    .map(|dopamine_buffer_item| {
                        self.get_weight_change_contrib(dopamine_buffer_item, &elig_trace)
                    })
                    .sum();

                PlasticityEvent {
                    weight_change,
                    pre_syn_nid: elig_trace.pre_syn_nid,
                    synapse_idx: elig_trace.synapse_idx,
                }
            })
            .filter(|event| match event.weight_change.partial_cmp(&0.0f32) {
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Less) => true,
                _ => false,
            })
    }
}

#[derive(Debug, Clone)]
struct DopamineBufferItem {
    t_release: usize,
    scaled_amount: f32,
}

impl DopamineBufferItem {
    fn new(t_release: usize) -> Self {
        Self {
            t_release,
            scaled_amount: 0.0,
        }
    }
}

impl PlasticityModulator {
    pub fn new(params: PlasticityModulationParams) -> Self {
        Self {
            elig_traces: VecDeque::new(),
            dopamine_buffer: Vec::new(),
            pending_dopamine_buffer_item: DopamineBufferItem::new(
                params.dopamine_conflation_period,
            ),
            next_dopamine_buffer_flush_t: params.dopamine_flush_period,
            params,
        }
    }

    pub fn process_stdp_value(
        &mut self,
        t: usize,
        pre_syn_nid: usize,
        synapse_idx: usize,
        stdp_value: f32,
    ) {
        self.elig_traces.push_back(EligibilityTrace {
            t: t + self.params.eligibility_trace_delay,
            stdp_value,
            pre_syn_nid,
            synapse_idx,
        });
    }

    pub fn process_dopamine(&mut self, amount: f32) {
        self.pending_dopamine_buffer_item.scaled_amount +=
            amount * self.params.dopamine_modulation_factor;
    }

    fn finalize_dopamine_buffer_item(&mut self) {
        self.dopamine_buffer
            .push(self.pending_dopamine_buffer_item.clone());

        self.pending_dopamine_buffer_item = DopamineBufferItem::new(
            self.pending_dopamine_buffer_item.t_release + self.params.dopamine_conflation_period,
        );
    }

    fn flush_dopamine_buffer(&mut self) -> PlasticityEvents<'_> {
        self.next_dopamine_buffer_flush_t += self.params.dopamine_flush_period;

        PlasticityEvents::new(
            &self.elig_traces,
            self.dopamine_buffer.drain(..).collect(),
            self.params.tau_eligibility_trace,
        )
    }

    pub fn tick(&mut self, t: usize) -> Option<PlasticityEvents<'_>> {
        if self.pending_dopamine_buffer_item.t_release == t {
            self.finalize_dopamine_buffer_item();
        }

        if self.next_dopamine_buffer_flush_t == t {
            self.drain_stale_elig_traces(self.dopamine_buffer.first().unwrap().t_release);
            Some(self.flush_dopamine_buffer())
        } else {
            None
        }
    }

    fn drain_stale_elig_traces(&mut self, t: usize) {
        loop {
            match self.elig_traces.front() {
                Some(elig_trace) if elig_trace.t + self.params.t_cutoff_eligibility_trace < t => {
                    self.elig_traces.pop_front()
                }
                _ => break,
            };
        }
    }
}

#[derive(Debug)]
struct EligibilityTrace {
    t: usize,
    stdp_value: f32,
    pre_syn_nid: usize,
    synapse_idx: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    fn tick_unwrap(sut: &mut PlasticityModulator, t: usize) -> Vec<PlasticityEvent> {
        sut.tick(t).unwrap().iter().collect()
    }

    #[test]
    fn zero_dopamine() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 1;

        let mut sut = PlasticityModulator::new(params);

        sut.process_stdp_value(0, 0, 1, 1.0);

        assert!(sut.tick(0).is_none());
        assert!(tick_unwrap(&mut sut, 1).is_empty());
    }

    #[test]
    fn dopamine_modulation_factor() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 1;
        params.eligibility_trace_delay = 1;
        params.dopamine_modulation_factor = 1.5;

        let mut sut = PlasticityModulator::new(params);

        sut.process_stdp_value(0, 0, 1, 1.2);

        assert!(sut.tick(0).is_none());

        sut.process_dopamine(0.5);

        let plasticity_events = tick_unwrap(&mut sut, 1);

        assert_eq!(plasticity_events.len(), 1);
        let event = plasticity_events.first().unwrap();
        assert_eq!(event.pre_syn_nid, 0);
        assert_eq!(event.synapse_idx, 1);
        assert_approx_eq!(f32, event.weight_change, 0.9);
    }

    #[test]
    fn eligbility_trace_cutoff_time() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 2;
        params.eligibility_trace_delay = 0;
        params.t_cutoff_eligibility_trace = 2;
        params.tau_eligibility_trace = 1.0;

        let mut sut = PlasticityModulator::new(params);

        sut.tick(0);

        sut.process_stdp_value(0, 0, 0, 1.0);

        assert!(sut.tick(1).is_none());
        sut.process_dopamine(1.0);
        sut.process_stdp_value(1, 1, 1, 1.0);

        // both eligility traces are before cutoff time -> two events
        let tick_2_events = tick_unwrap(&mut sut, 2);

        let event_0 = tick_2_events.get(0).unwrap();

        assert_eq!(tick_2_events[0].pre_syn_nid, 0);
        assert_eq!(tick_2_events[0].synapse_idx, 0);
        assert_approx_eq!(
            f32,
            event_0.weight_change,
            util::get_decay_factor(2, 0, 1.0)
        );

        assert_eq!(tick_2_events[1].pre_syn_nid, 1);
        assert_eq!(tick_2_events[1].synapse_idx, 1);
        assert_approx_eq!(
            f32,
            tick_2_events[1].weight_change,
            util::get_decay_factor(2, 1, 1.0)
        );

        assert!(sut.tick(3).is_none());
        sut.process_dopamine(1.0);

        // only one eligibility trace remaining now -> one event
        let tick_4_events = tick_unwrap(&mut sut, 4);

        assert_eq!(tick_4_events.len(), 1);

        let event_0 = tick_4_events.first().unwrap();
        assert_eq!(event_0.pre_syn_nid, 1);
        assert_eq!(event_0.synapse_idx, 1);
        assert_approx_eq!(
            f32,
            event_0.weight_change,
            util::get_decay_factor(4, 1, 1.0)
        );
    }

    #[test]
    fn dopamine_conflation_and_flush() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 2;
        params.dopamine_flush_period = 4;
        params.eligibility_trace_delay = 0;
        params.t_cutoff_eligibility_trace = 10;
        params.tau_eligibility_trace = 1.0;

        let mut sut = PlasticityModulator::new(params);

        sut.tick(0);

        sut.process_dopamine(1.0);
        sut.process_stdp_value(0, 0, 0, 1.1);
        sut.tick(1);
        sut.process_dopamine(1.0);

        assert!(sut.dopamine_buffer.is_empty());
        assert_approx_eq!(f32, sut.pending_dopamine_buffer_item.scaled_amount, 2.0);
        assert!(sut.tick(2).is_none());

        assert_eq!(sut.dopamine_buffer.len(), 1);
        assert_eq!(sut.dopamine_buffer.first().unwrap().t_release, 2);
        assert_approx_eq!(f32, sut.dopamine_buffer.first().unwrap().scaled_amount, 2.0);
        assert_approx_eq!(f32, sut.pending_dopamine_buffer_item.scaled_amount, 0.0);

        sut.process_dopamine(2.0);
        sut.tick(3);
        sut.process_dopamine(1.0);

        let events = tick_unwrap(&mut sut, 4);
        assert_eq!(events.len(), 1);
        assert_approx_eq!(
            f32,
            events[0].weight_change,
            1.1 * (2.0 * (-2.0f32).exp() + 3.0 * (-4.0f32).exp())
        );

        assert!(sut.dopamine_buffer.is_empty());
        assert_eq!(sut.elig_traces.len(), 1);
        assert_approx_eq!(f32, sut.pending_dopamine_buffer_item.scaled_amount, 0.0);
    }

    #[test]
    fn eligibility_trace_delay() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 1;
        params.eligibility_trace_delay = 2;
        params.dopamine_modulation_factor = 1.0;
        params.tau_eligibility_trace = 1.0;

        let mut sut = PlasticityModulator::new(params);

        sut.tick(0);
        sut.process_dopamine(1.0);
        sut.tick(1);
        sut.process_stdp_value(1, 0, 0, 1.0);

        sut.process_dopamine(1.0);
        assert!(tick_unwrap(&mut sut, 2).is_empty());

        sut.process_stdp_value(2, 1, 1, 1.0);

        sut.process_dopamine(1.0);

        let tick_3_events = tick_unwrap(&mut sut, 3);

        // first trace delay is past, second trace delay not yet
        assert_eq!(tick_3_events.len(), 1);
        assert_eq!(tick_3_events[0].pre_syn_nid, 0);
        assert_eq!(tick_3_events[0].synapse_idx, 0);
        assert_approx_eq!(f32, tick_3_events[0].weight_change, 1.0);

        sut.process_dopamine(1.0);
        let tick_4_events = tick_unwrap(&mut sut, 4);

        // now both delays are past
        assert_eq!(tick_4_events.len(), 2);
        assert_eq!(tick_4_events[0].pre_syn_nid, 0);
        assert_eq!(tick_4_events[0].synapse_idx, 0);
        assert_approx_eq!(f32, tick_4_events[0].weight_change, 1.0 * (-1.0f32).exp());
        assert_eq!(tick_4_events[1].pre_syn_nid, 1);
        assert_eq!(tick_4_events[1].synapse_idx, 1);
        assert_approx_eq!(f32, tick_4_events[1].weight_change, 1.0);
    }

    #[test]
    fn negative_eligibility_trace_decay() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 1;
        params.eligibility_trace_delay = 0;
        params.dopamine_modulation_factor = 1.0;
        params.tau_eligibility_trace = 10.0;

        let mut sut = PlasticityModulator::new(params);

        sut.tick(0);
        sut.process_stdp_value(0, 0, 0, -2.0);
        sut.process_dopamine(1.0);

        let events = tick_unwrap(&mut sut, 1);

        assert_approx_eq!(f32, events[0].weight_change, -2.0 * (-0.1f32).exp());
    }

    #[test]
    fn negative_dopamine_amount() {
        let mut params = PlasticityModulationParams::default();
        params.dopamine_conflation_period = 1;
        params.dopamine_flush_period = 1;
        params.eligibility_trace_delay = 0;
        params.dopamine_modulation_factor = 1.0;
        params.tau_eligibility_trace = 10.0;

        let mut sut = PlasticityModulator::new(params);

        sut.tick(0);
        sut.process_stdp_value(0, 0, 0, 2.0);
        sut.process_dopamine(-1.0);

        let events = tick_unwrap(&mut sut, 1);
        assert_approx_eq!(f32, events[0].weight_change, -2.0 * (-0.1f32).exp());
    }

    #[test]
    fn complex_scenario() {
        // testing a complex scenario against manually calculated values

        let params = PlasticityModulationParams {
            tau_eligibility_trace: 800.0,
            eligibility_trace_delay: 100,
            dopamine_modulation_factor: 0.9,
            t_cutoff_eligibility_trace: 2000,
            dopamine_flush_period: 1000,
            dopamine_conflation_period: 100,
        };

        let mut sut = PlasticityModulator::new(params);

        for t in 0..601 {
            assert!(sut.tick(t).is_none());
        }

        sut.tick(601);
        sut.process_stdp_value(601, 1, 2, -0.2);

        for t in 601..650 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(0.1);

        for t in 650..850 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(0.1);
        assert!(sut.tick(850).is_none());
        sut.process_dopamine(0.2);

        for t in 851..1000 {
            assert!(sut.tick(t).is_none());
        }

        let events_1s = tick_unwrap(&mut sut, 1000);

        assert_eq!(events_1s.len(), 1);

        assert_eq!(events_1s[0].pre_syn_nid, 1);
        assert_eq!(events_1s[0].synapse_idx, 2);
        assert_approx_eq!(f32, events_1s[0].weight_change, -0.0421078442);

        for t in 1001..1500 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(0.5);

        for t in 1500..1901 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_stdp_value(1900, 3, 4, 0.4);

        for t in 1901..1950 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(0.6);

        for t in 1950..2000 {
            assert!(sut.tick(t).is_none());
        }

        let events_2s = tick_unwrap(&mut sut, 2000);

        assert_eq!(events_2s.len(), 2);

        assert_eq!(events_2s[0].pre_syn_nid, 1);
        assert_eq!(events_2s[0].synapse_idx, 2);
        assert_approx_eq!(f32, events_2s[0].weight_change, -0.05444362263);

        assert_eq!(events_2s[1].pre_syn_nid, 3);
        assert_eq!(events_2s[1].synapse_idx, 4);
        assert_approx_eq!(f32, events_2s[1].weight_change, 0.216);

        for t in 2001..2650 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(1.0);

        for t in 2650..3000 {
            assert!(sut.tick(t).is_none());
        }

        let events_3s = tick_unwrap(&mut sut, 3000);

        assert_eq!(events_3s.len(), 2);

        assert_eq!(events_3s[0].pre_syn_nid, 1);
        assert_eq!(events_3s[0].synapse_idx, 2);
        assert_approx_eq!(f32, events_3s[0].weight_change, -0.01479378042);

        assert_eq!(events_3s[1].pre_syn_nid, 3);
        assert_eq!(events_3s[1].synapse_idx, 4);
        assert_approx_eq!(f32, events_3s[1].weight_change, 0.15007032708);

        for t in 3001..3500 {
            assert!(sut.tick(t).is_none());
        }

        sut.process_dopamine(10.0);

        for t in 3500..4000 {
            assert!(sut.tick(t).is_none());
        }

        assert_eq!(sut.elig_traces.len(), 2);
        let events_4s = tick_unwrap(&mut sut, 4000);
        assert_eq!(sut.elig_traces.len(), 1);
        assert_eq!(events_4s.len(), 1);

        assert_eq!(events_4s[0].pre_syn_nid, 3);
        assert_eq!(events_4s[0].synapse_idx, 4);
        assert_approx_eq!(f32, events_4s[0].weight_change, 0.55207788064);

        sut.process_dopamine(10.0);

        for t in 4001..5000 {
            assert!(sut.tick(t).is_none());
        }

        let events_5s = tick_unwrap(&mut sut, 5000);
        assert!(events_5s.is_empty());
        assert!(sut.elig_traces.is_empty());
    }
}
