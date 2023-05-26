#[derive(Debug, Clone, Copy)]
pub enum SCMode {
    Off,
    Single { threshold: f32 },
    Multi { threshold: f32 },
}
