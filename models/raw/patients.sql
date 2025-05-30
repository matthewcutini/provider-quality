SELECT
id as patient_id,
birthdate as birth_date,
deathdate as death_date,
ssn,
drivers as driver_license,
prefix,
first as first_name,
last as last_name,
marital as marital_status,
race,
ethnicity,
gender,
address,
state,
zip,
lat,
lon
FROM {{ source('synthea', 'patients') }}