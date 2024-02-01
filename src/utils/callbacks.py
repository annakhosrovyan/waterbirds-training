from pytorch_lightning.callbacks import Callback

class PrintingCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("\nStarting to train!\n")

    def on_train_end(self, trainer, pl_module):
        print("\nTraining is done.\n")
