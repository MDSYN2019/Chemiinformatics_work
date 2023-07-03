SELECT id, structure
FROM molecules
WHERE structure@>'C1=CC=C(C=C1)[CH-]C1=CC=CC=C1.[Fe+2]' -- find structures which include ferrocene
