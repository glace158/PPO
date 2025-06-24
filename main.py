import sys

from ppo_run import TrainRun, TestRun

if __name__ == '__main__':
    
    arg = sys.argv

    if len(arg)<= 1:        
        arg = ["main.py", "test", "./PPO_preTrained/Humanoid-v5/PPO_Humanoid-v5_0_20250624-131950.pth"]
        #arg = ["main.py", "train", ""]
        

    if len(arg) > 1:
        if arg[1] == 'train':
            richdog = TrainRun(checkpoint_path=arg[2])
            richdog.random_train()
        elif arg[1] == 'test':
            richdog = TestRun(checkpoint_path=arg[2])
            richdog.test()
