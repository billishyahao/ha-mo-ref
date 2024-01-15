import matplotlib.pyplot as plt

file = open('hl-smi-gpt-fp8.log', 'r')
lines = file.readlines()

powers =[]
powers_in_nodes =[]
for index,line in enumerate(lines):
    if "HL-225C" in line:
        next_line = lines[index+1]
        powers_in_nodes.append(int(next_line.split("   ")[3].split("W")[0]))
        if len(powers_in_nodes)==8:
            powers.append(sum(powers_in_nodes))
            powers_in_nodes =[]

print(sum(powers)/len(powers))
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(powers) + 1), powers, linestyle='-')
plt.xlabel('Times')
plt.ylabel('Power(W)')
plt.title('Power over Times')
plt.grid(True)
plt.tight_layout()
plt.savefig('fp8.png')

