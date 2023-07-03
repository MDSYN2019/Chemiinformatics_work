SELECT id, structure
FROM molecules
WHERE structure@>'c1ccccc1' -- find structures which include benzne 
