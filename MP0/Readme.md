
## Objective

The purpose of this lab is to get you familiar with using the submission system for this course and the hardware used.

## Instructions

Click on the code tab and then read the code written.
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

* GPU card's name

* GPU computation capabilities

* Maximum number of block dimensions

* Maximum number of grid dimensions

* Maximum size of GPU memory

* Amount of constant and share memory

* Warp size

All results from previous attempts can be found in the Attempts tab.
You can choose any of these attempts for submission for grading.
Note that even though you can submit multiple times, only your last submission will be reflected in the Coursera database.

After completing this lab, and before proceeding to the next one, you will find it helpful to read the [tutorial](/help) document

