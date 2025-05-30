SELECT
`start` as start_date,
`stop` as end_date,
patient as patient_id,
encounter as encounter_id,
code as condition_code,
`description` as condition_description,
current_date as last_updated_date
from {{ source('synthea', 'conditions') }}