module Main where

import Torch.Data.ActionTransitionSystem (testProgram)

main :: IO ()
main = testProgram 0.0005 1000 10 1 "model.pt"
