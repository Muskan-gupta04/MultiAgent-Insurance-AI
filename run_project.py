from src.data_pipeline import initialize_data_infrastructure
from src.workflow import run_test_query


def main():
    initialize_data_infrastructure()
    run_test_query("In general, what does life insurance cover?")


if __name__ == "__main__":
    main()
