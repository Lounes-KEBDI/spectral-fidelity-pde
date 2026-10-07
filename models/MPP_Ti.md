# MPP-Ti weights

The MPP-Ti weights are **not on Hugging Face**. They are in a Google Drive folder named
**"MPP Pretrained Weights"**, linked from the README of
[PolymathicAI/multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining):

https://drive.google.com/drive/folders/1Qaqa-RnzUDOO8-Gi4zlf4BE53SfWqDwx

## How to get them

1. Open the Google Drive folder above.
2. Find the file for the **Ti** model.
3. Download it into `models/MPP-Ti/`, either:
   - **by hand** from the browser, or
   - **with `gdown`** (`pip install gdown`), using the file's link:

     ```bash
     gdown "<file link>" -O models/MPP-Ti/
     ```

## Notes

- The exact file name has **not been verified**.
- According to the MPP README, the model configuration (Ti / S / B / L) must match the
  downloaded weights file. Use the **Ti** configuration with these weights.
