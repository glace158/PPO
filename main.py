import sys

from ppo_run import TrainRun, TestRun

if __name__ == '__main__':
    
    arg = sys.argv

    if len(arg)<= 1:        
        #arg = ["main.py", "train", "PPO_preTrained/MountainCarContinuous-v0/PPO_MountainCarContinuous-v0_0_20250514-224951.pth"]
        arg = ["main.py", "train", ""]
        

    if len(arg) > 1:
        if arg[1] == 'train':
            richdog = TrainRun()
            richdog.random_train()
        elif arg[1] == 'test':
            richdog = TestRun(checkpoint_path=arg[2])
            richdog.test()
