# 'w' is defined here and passed as argument for the events
# IF AFTER EXECUTING a returning task is not completed, it was discarded 
# create a time dictionary of communication so it doesn't need to do math every time

# 1. generate a first round of recieving tasks
# 2. run events that generate more events
# 3. check a state and make decisions
# 4. generate another round of recieving based on the new 'w' and repeat from 2.