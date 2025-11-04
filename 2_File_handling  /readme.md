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
