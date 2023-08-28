## Lab0 (MP0) Instructions

The purpose of this lab is to get you familiar with using the submission system for this course and the hardware used.

Once you setup rai correctly, you can run lab0 (MP0) with the following command (assume you are in the parent directory):
```bash
rai -p ./MP0
```

If your `.rai_profile` is correct, you will see the following lines printed to your terminal:

    ✱ Checking your authentication credentials.
    ✱ Preparing your project directory for upload.
    ✱ Uploading your project directory. This may take a few minutes.

When the server starts to process your submission, you will see the compilation and execution messages on your terminal. In this lab, we simply fetch the CUDA device information with `cudaGetDeviceProperties`. You will see the following entries and corresponding data on your terminal. You will need these data to complete the lab0 quiz.

* GPU card's name

* GPU computation capabilities

* Maximum global memory size

* Maximum constant memory size

* Maximum shared memory size per block

* Maximum block dimensions

* Maximum grid dimensions

* Warp size

After the server finishes executing your job, it will pack your build folder and save it to an URL. You will see a message similar as below:

```bash
 ✱ The build folder has been uploaded to http://your_url.tar.gz. The data will be present for only a short duration of time.
```

You will need this URL to complete the first question of your lab0 quiz. The quiz is autograded, and the URL is saved for regrading purpose. Make sure to answer each question before turning in the quiz.  

