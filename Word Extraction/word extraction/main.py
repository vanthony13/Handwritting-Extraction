import matplotlib.pyplot as plt

# set background color
plt.style.use('dark_background')
fig = plt.figure(facecolor='#13120b')

x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y1 = [0, 0.12, 0.38, 0.5, 0.63, 0.71, 0.77, 0.81, 0.84, 0.81, 0.79]
y2 = [0, 0.14, 0.43, 0.50, 0.57, 0.61, 0.64, 0.66, 0.71, 0.69, 0.71]
y3 = [0, 0.11, 0.2, 0.37, 0.51, 0.59, 0.60, 0.58, 0.60, 0.57, 0.59]

# remove markers and set color for each line
plt.plot(x, y1, '-', color='red', label='EasyOCR')
plt.plot(x, y2, '-', color='green', label='KerasOCR')
plt.plot(x, y3, '-', color='blue', label='PyTesseract')

# set x and y axis labels
plt.xlabel('Percentage of characters recognized')
plt.ylabel('Percentage of words correctly recognized')

# set x and y ticks
plt.xticks(x)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# set grid color
plt.grid(color='#5a5a5a')

# show legend
plt.legend()

# show plot
plt.show()
