import argparse
import sys
import requests

def main():
    parser = argparse.ArgumentParser(description='CDIE v5 Health Probe')
    parser.add_argument('--url', required=True, help='Health endpoint URL')
    parser.add_argument('--timeout', type=int, default=5, help='Request timeout')
    args = parser.parse_args()

    try:
        response = requests.get(args.url, timeout=args.timeout)
        if response.status_code == 200:
            print(f"Health check PASSED for {args.url}")
            sys.exit(0)
        else:
            print(f"Health check FAILED for {args.url} (Status: {response.status_code})")
            sys.exit(1)
    except Exception as e:
        print(f"Health check ERROR for {args.url}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
