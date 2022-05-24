# Python program to demonstrate
# command line arguments


import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-o", "--Output", help = "Show Output")
parser.add_argument("-n", "--Normalization", help = "Normal")

# Read arguments from command line
args = parser.parse_args()

print("Displaying Output as: % s" % args.Normalization)

if args.Output:
	print("Displaying Output as: % s" % args.Output)
