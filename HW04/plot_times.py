import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('consolidated_data.csv')

# Plot the data with different markers and colors
plt.figure(figsize=(10, 6))
plt.plot(
    df['Number of threads'], df['Time_Dynamic'],
    marker='o', color='blue', label='Dynamic'
)
plt.plot(
    df['Number of threads'], df['Time_Guided'],
    marker='s', color='green', label='Guided'
)
plt.plot(
    df['Number of threads'], df['Time_Static'],
    marker='^', color='red', label='Static'
)

# Add labels and title
plt.xlabel('Number of Threads')
plt.ylabel('Time (ms)')
plt.title('Number of Threads vs Time')
plt.legend()
plt.grid(True)

# Save the plot as a PDF file
plt.savefig('threads_vs_time.pdf')

# Show the plot
plt.show()
