import matplotlib.pyplot as plt

# import tools.data_oranizer

str = "abcdefg"
with open("result.txt", "w") as f:
    str = str[1:-1]

    f.write(str)
