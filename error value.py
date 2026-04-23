import numpy as np

# মনে করুন আপনার ১০টি আলাদা ভিডিও সিকোয়েন্সের Accuracy রেজাল্ট এগুলো:
custom_results = [67.41, 81.67, 66.96, 55.64, 53.18]

# Standard Error বের করার কোড:
mean = np.mean(custom_results)
std_dev = np.std(custom_results)
standard_error = std_dev / np.sqrt(len(custom_results))

print(f"গড় (Mean): {mean}")
print(f"এরর মান (Error Value): {standard_error}")