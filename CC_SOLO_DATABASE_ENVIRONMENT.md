# Local database validation environment

The source baseline is petl v1.7.17 plus reproducible setup documentation.
Database evaluation uses Python 3.11, standard-library sqlite3, and SQLAlchemy
1.4.54 in its legacy (non-future) API mode. No external database service,
credentials, real business data, or SQLAlchemy 2.x migration is required.

Follow `CC_SOLO_ENVIRONMENT.md` to bootstrap pip, then install the database set:

```sh
python3 -m pip install --user --break-system-packages -r requirements-cc-solo-db.txt
python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q petl/test/io/test_db.py petl/test/io/test_sqlite3.py petl/test/io/test_db_create.py petl/test/io/test_db_inference.py
```

Use temporary local SQLite databases for real transaction and persistence
checks. Recording DB-API doubles are appropriate for driver-specific SQL,
parameter styles, delayed cursor descriptions, and injected lifecycle errors;
identify them as doubles, not external-server validation.
