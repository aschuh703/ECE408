# ECE408 Fall 2022

Welcome to ECE408 / CS483 / CSE408 Fall 2022!

Checkout the [Introduction to RAID](https://drive.google.com/file/d/1t6-uPgbCxi5zx0FKKG15nanXt7NX8zCP/view?usp=sharing) PDF. This document explains the RAI/RAID organization and options. Note, some of the more advanced options like using an exclusive queue will not be available until it is time to work on the project.

# Install and Setup

Clone this repository to get the project folder.

    git clone https://github.com/aschuh703/ECE408.git


Download the RAI binary for your platform below.

[Linux and Darwin RAI binaries](https://drive.google.com/drive/folders/1Pp84x3So9OEHUwRHQVZcRP441wRsO-UV)

In the documentation below, we refer to RAI as the RAI binary. Either rename your binary to `rai` or substitute the downloaded name when executing. When executing, you will need to include the full path to RAI or add RAI's location to your `$PATH`.

Finally, the downloaded file will not have execute privileges. Use the `chmod +x rai` command to make the file executable.

You should have received a `.rai_profile` file by email. Do not share your .rai_profile with anyone. 
Put that file in `~/.rai_profile`.
Your `.rai_profile` should look something like this (indented with spaces!). The following profile is just a sample. Your actual profile may contain other fields like team and role.

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: <your-affiliation>

To test the configurations, execute RAI with the lab0 (MP0). For example, if you are in the parent directory of this repo, you will run lab0 (MP0) like:

```bash
/your/path/to/rai/rai -p ./MP0
```
Alternatively, if you have added RAI's location to your `$PATH`, simply run:
```bash
rai -p ./MP0
```

### Windows

****
On Windows, you'll need to install WSL and a virtual linux OS. Several Linux versions are available
through the Microsoft Store.
