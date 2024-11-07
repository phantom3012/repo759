import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# File names
files = ['OUTPUTS.TXT', 'OUTPUTS_16.TXT']
labels = ['512 threads', '16 threads']
markers = ['o', 's']
colors = ['red', 'blue']

# Initialize data storage
data = {label: {'size': [], 'time': []} for label in labels}

# Read each file and store the data
for file, label in zip(files, labels):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 5):  # Adjusting to handle the newline
            # Ensure lines are not empty
            if (lines[i].strip() and lines[i + 1].strip()):
                size = float(lines[i].strip())
                time = float(lines[i + 1].strip())
                data[label]['size'].append(size)
                data[label]['time'].append(time)

# Plot the data
plt.figure(figsize=(12, 6))  # Set the figure size to be square
for label, marker, color in zip(labels, markers, colors):
    plt.plot(data[label]['size'], data[label]['time'],
             marker=marker, color=color, label=label)
    plt.plot(data[label]['size'], data[label]['time'],
             linestyle='-', color=color)  # Add lines connecting the points

# Add labels and title
plt.xlabel('Size of Array')
plt.ylabel('Time (in milliseconds)')
plt.title('Scaling Analysis')
plt.legend()
plt.grid(True, which="both", ls="--")

# Set x-ticks to be exactly the sizes of the arrays
all_sizes = sorted(set(data['512 threads']['size'] +
                       data['16 threads']['size']))
plt.xticks(all_sizes, rotation=45, ha='right', fontsize=10)

# Set the x-axis to logarithmic scale with base 2
plt.xscale('log', base=2)


# Custom formatter to label x-axis as powers of 2
def log2_formatter(x, pos):
    return f'$2^{{{int(np.log2(x))}}}$'


plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(log2_formatter))
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())

# Ensure all sizes are included in the major ticks
plt.gca().xaxis.set_major_locator(ticker.FixedLocator(all_sizes))

# Adjust the layout to make room for the rotated x-ticks
plt.tight_layout(pad=2.0)

# Save the plot as a PNG file
plt.savefig('time_vs_size.png')

# Show the plot
plt.show()
