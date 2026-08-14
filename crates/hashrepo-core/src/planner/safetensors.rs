use super::{ByteSource, Plan, PlanError};

pub fn try_plan(_source: &dyn ByteSource) -> Result<Option<Plan>, PlanError> {
    Ok(None)
}
