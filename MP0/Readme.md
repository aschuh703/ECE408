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

You will need this URL to complete the first question of your lab0 quiz. In this lab0 quiz, all questions will be immediately graded except the first one, so please make sure you have filled in all the answers before submitting.


<!-- Click on the code tab and then read the code written.
Do not worry if you do not understand all the details of the code (the purpose is to get you familiar with the submission system).
Once done reading, click the "Compile & Run" button.

The submission system will automatically switch to the compile-and-run results that will also be available through the **Attempts** tab.
There, you will be able to see a summary of your attempt.

The `Timer` section has 3 columns:

* *Kind* corresponds with the first argument to `wbTimer_start`,
* *Location* describes the `file::line_number` of the `wbTimer` call, and
* *Time* in millisecond that it took to execute the code in between the `wbTime_start` and `wbTime_stop`, and
* *Message* the string you passed into the second argument to the timer

Similarly, you will see the following information under the `Logger` section.

The `Logger` section has 3 columns:

* *Level* is the level specified when calling the `wbLog` function (indicating the severity of the event),
* *Location* describes the `function::line_number` of the `wbLog` call, and
* *Message* which is the message specified for the `wbLog` function

The `Timer` or `Logger` seconds are hidden, if no timing or logging statements occur in your program.

We log the hardware information used for this course --- the details which will be explained in the first few lectures.

All results from previous attempts can be found in the Attempts tab.
You can choose any of these attempts for submission for grading.
Note that even though you can submit multiple times, only your last submission will be reflected in the Coursera database.

After completing this lab, and before proceeding to the next one, you will find it helpful to read the [tutorial](/help) document -->

