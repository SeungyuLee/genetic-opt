import network
import sys

def main():
	for i in range(1, 5):
		for j in range(1, 5):
			for k in range(1, 5):
				for l in range(1, 5):
					for m in range(1, 5):
						network.execute(i*10+60, j*10+30, k*10, l/1.0, m/10.0, "brute_result.txt")

	return None

if __name__ == "__main__":
	main()
