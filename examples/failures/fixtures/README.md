# fixtures/

Harvested failure logs land here, one per failure class:

- `<nn-name>.log` — the job's `logs/<job_name>_<job_id>.out`
- `exitcodes.txt` — the `sacct -o JobID,State,ExitCode,Elapsed --parsable2` line per job

Later these move into `tests/fixtures/` as the seed corpus for the debug
command's signature library.
