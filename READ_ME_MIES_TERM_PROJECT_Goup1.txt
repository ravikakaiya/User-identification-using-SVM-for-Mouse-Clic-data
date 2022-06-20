@....Importing neccessary libraries 

@....spliting of raw text data into list of lines :   1. line.rstrip()---- for creating a list of wach new line in text data  
                                                      2. .split() --- for spliting lines into list of words separated by space

@.....function--- def timediff(t1,t2)             :   1. datetime.striptime()-----  used to convert date into our required format of date/month/year/hour/minute/second                                                           2.  .totalseconds() --------- used to calculate the time difference between t1 and t2.

@....separating the line depending on type of mouse movements ie. MM / MWM/ MP/MR  : 1.  we have defined 8 list(2 sets of 4 list) to separate 4 different mouse movement ,                                                                                            also to differentiate between the fake and real user . for fake users we                                                                                                    incorporated all users other than the real user (in our case  we had data from                                                                                              three user )
                                                                                     2.  scr / scr1 stores the no. of wheel movements in that time interval
                                                                                     3.  xmr / ymr are the pixels coordimate for real user  while xmr2/ymr2 are the pixels                                                                                          coordinates for fake users.
                                                                                     4.  tmr/ tmr1 are the time instants for real and fake users for MM while tmwr/tmwr1                                                                                             are the time instants for fake users 

@-----function --- def angular_motion(scr,tmw)     : returns the values of angular velocity betweeen MWM.

@-----function----def  fet_values(xmmr,tmr,ymmr)   : returns the time difference and the distance between 2 data points

@-----defining batch size = 1000(we took this in our case)   

@-----defining nbd which define number of batches.

@-----features extraction----def mean_std(db,tb,nbd)  :  finds mean and standrad deviation of quantities batchwise ie. for each batch of batchsize = 1000 this fuction                                                          returns a mean and standard deviation assosiated with each batch {db/tb are nothing but 2 quantities for which                                                          mean and standard deviation to be calculated}. so basically our features that will be used to train a classifier                                                          will be mean and standard deviation of quantities (like --distance travelled , time difference , acceleration ,                                                          velocity ,angular velocity)

@-----train/test split   : we splitted the data we got from group_2 in test and train data in ratio of 20:80 ie. 80% of the data will be used to train the classifier.

@-----using matplotlib : we tried to plot for different features that could define a user or could be used to identify characterstics of user . from this we took 2                          features simultaneously and tried to plot using matplotlib for week 1 / week1+week2/week1+week2+wee3/consolidated data to see which gives less                          overlap and also checked the validation accuracy to come to a conclusion to use {mean of angular velocity of a batch / mean of distance travelled                          for a batch

@----using scikit learn inbuilt svm classifier : as per our given problem statement we have to use svm for classification of users . after hit and trial we came to                                                  conclusion that radial basis function is most suited for our user classification task . Then we used train data(80% of                                                  total) to train our svm classifier and finally fit it.

@-----using scikit learn inbuilt evaluation metric to calculate the score of classification using k fold validation

@-----finally using 20% of total data to test the fidelity of our model using accuracy metric.

@-----for running this code :      1. !pip instll python-math
                                   2. !pip install os-sys
                                   3. !pip install strings
                                   4. !pip install DateTime
                                   5. !pip install numpy
                                   6. !pip install more-itertools
                                   7. keep the merged text file in the directory as mentioned under fopen("same as the directory where u stored the merged text file")

@----- for running classification code in jupyter notebook we have attached a .ipynb file with the delievarables.






