import numpy as np
from matplotlib import pyplot as plt
class DiscreteGenerator:
  def __init__(self, variables_to_generate, probabilities):
    # Discrete Variables
    self.variables_to_generate = variables_to_generate

    # Desired probabilities of the discrete variables
    self.probabilities = probabilities

    # Average of probabilities
    a = 1/len(variables_to_generate)

    # FINDING K and V
    # Initialize
    self.k = np.array([variables_to_generate[i] for i in range(len(variables_to_generate))], dtype="U256")
    self.v = np.array([(i+1)*a for i in range(len(variables_to_generate))], dtype="float32")

    height = np.copy(probabilities)
    for it in range(len(variables_to_generate)-1):
      #Min and max
      pi, i = height.min(), np.argmin(height)
      pj, j = height.max(), np.argmax(height)
      self.k[i] = variables_to_generate[j]
      self.v[i] = (i)*a + pi
      height[j]= pj-(a-pi)
      height[i]=a

  def print_hist(self):
    # For explainability
    n = len(self.variables_to_generate)
    h = 1/len(self.variables_to_generate)
    hc = int(h*100)
    wc = 5
    columns = []
    for j in range(n):
      frontier = np.ceil((np.round(self.v[j]- j*h, 2)*100)).astype('int')
      row = [wc*self.variables_to_generate[j]] * frontier
      krow = [wc * self.k[j]] * (hc - frontier)
      rows = krow + row
      columns.append(np.reshape(rows, (hc, 1)))
    
    histogram = np.hstack(columns)
    print(histogram)

  def generate_value(self):
    U = np.random.rand()
    j = np.floor(U*5).astype('int')
    return self.variables_to_generate[j] if U < self.v[j] else self.k[j]

generator = DiscreteGenerator(["A", "B", "C", "D", "E"], [.21,.18,.26,.17,.18])
generator.print_hist()

# Check reliability

start = 100
end = 10000
step = 10
n_samples = np.linspace(start, end, step)
results = []
for samples in n_samples:
  data = np.array([generator.generate_value() for i in range(int(samples))])
  aux = np.array([np.char.count(data, c).sum() for c in ["A", "B", "C", "D", "E"]])/samples
  results.append(aux)

results = np.array(results)
group_by = 10
interval = int(results.shape[0]/group_by)
grouped = np.array([results[interval*i:interval*(i+1),:].mean(axis=0) for i in range(group_by)])
errors = np.array([((res - [.21,.18,.26,.17,.18]) ** 2).mean() for res in grouped])

plt.plot(np.linspace(start, end, errors.shape[0]),errors)
plt.title("Error per number of Samples taken")
plt.xlabel("Number of samples")
plt.ylabel("Mean Squared Error")
plt.show()