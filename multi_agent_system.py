from mas.graph import build_app
from mas.run import init_all, run_test_query


__all__ = [
    "init_all",
    "run_test_query",
    "build_app",
]


if __name__ == "__main__":
    init_all(setup_db=False, ingest_faq=False)
    app = build_app()
    run_test_query("What is the premium of my auto insurance policy?", app=app)
