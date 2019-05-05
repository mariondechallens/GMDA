import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GMDA Project Marion Favre dEchallens and Laurent Lin')
    parser.add_argument('--ex', default=2, type=int, help='Choose the exercise number')
    args = parser.parse_args()

    assert args.ex in [2, 3, 4, 5], ValueError("Only 2, 3, 4, 5 are accepted values")

    os.system(f"python ex{args.ex}.py")

    print("Done")
