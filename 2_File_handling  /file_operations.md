

## ✅ **Python Code: File & File System Basics**

```python
# list files in current directory
import os
print("Files in directory:", os.listdir("."))

# get current working directory
print("Current path:", os.getcwd())

# check if a file exists
file_path = "example.txt"
print("Exists?", os.path.exists(file_path))

# read a text file
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    print("\nFirst 200 chars of example.txt:\n", content[:200])
else:
    print("\nexample.txt not found!")

# Read a CSV file (assume 'data.csv' exists in same folder)
import csv

csv_file = "data.csv"
if os.path.exists(csv_file):
    print(f"\nReading {csv_file}...\n")
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Header:", header)
        for row in reader:
            print(row)
            break  # print only first row for demo

    # Write to another CSV
    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "marks"])
        writer.writerow(["Swapnil", 95])
        writer.writerow(["Padma", 98])
    print("\n✅ output.csv written.")
else:
    print(f"\n{csv_file} not found!")

# append to a log file
with open("log.txt", "a") as f:
    f.write("Run completed.\n")

print("\n✅ Logging complete.")
```

---

## 📘 **Markdown Explanation**

### 🔍 Files & Directories

```python
os.listdir(".")        # shows files in current folder
os.getcwd()            # current working directory
os.path.exists(path)   # check if file/folder exists
```

### 📖 Reading Text File

```python
with open("example.txt", "r") as f:
    content = f.read()
```

* `r` → read mode
* Always use `with` to auto-close files

### 📂 CSV File Handling

#### ✅ Read CSV

```python
import csv
with open("data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)  # first row
```

#### ✨ Write CSV

```python
with open("output.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow([...])
```

### 📝 Append to log

```python
with open("log.txt","a") as f:
    f.write("Run completed\n")
```

### 📎 Notes

| Mode      | Meaning                          |
| --------- | -------------------------------- |
| `r`       | Read                             |
| `w`       | Write (overwrite)                |
| `a`       | Append                           |
| `rb`/`wb` | Read/write binary (images, PDFs) |

---

## 🎯 Your folder after running this:

```
example.txt
data.csv
output.csv   ← created by script
log.txt      ← appended
```

---
ant?
