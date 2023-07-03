SELECT id, structure
FROM molecules
WHERE structure@>'c1cccnc1' -- find structures which include pyridine
