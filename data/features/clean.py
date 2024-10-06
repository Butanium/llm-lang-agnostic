import os


def replace_string_in_files(directory, old_string, new_string):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()

            content = content.replace(old_string, new_string)

            with open(file_path, "w") as f:
                f.write(content)


# Replace "s1" with "s2" in all files within the specified directory and its subdirectories
directory = "/dlabscratch1/cdumas/thinking-lang/data/features"
old_string = ""
new_string = ""
replace_string_in_files(directory, old_string, new_string)
