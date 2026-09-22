import numpy as np
import pandas as pd
from banditpy.models import BanditTrainer2Arm
from banditpy.core import Bandit2Arm
from banditpy.utils import generate_probs_2arm
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed


class BatchRunModels2Arm:
    """A class to create and train multiple 2-arm bandit models with structured and unstructured training data

    Folder structure:
    - basedir/
        - Train{n_train_sessions}Test{n_test_sessions}Impure{frac_impurity}_{timestamp}/
            - structured_2arm_model{name_suffix}
                - structured_2arm_model{name_suffix}.pt  # Model file
                - structured_2arm_model{name_suffix}_structured  # Struc Data folder
                    - structured_2arm_model{name_suffix}_structured.csv  # Struc data
                - structured_2arm_model{name_suffix}_unstructured  # Unstruc Data folder
                    - structured_2arm_model{name_suffix}_unstructured.csv  # Unstruc data


    """

    def __init__(
        self,
        n_train_sessions=50000,
        n_test_sessions=500,
        frac_impurity=0.16,
        basedir="/mnt/pve/Homes/bapun/Data/rnn_data/",
    ):
        self.n_train_sessions = n_train_sessions
        self.n_test_sessions = n_test_sessions
        self.frac_impurity = frac_impurity

        self.probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        self.unstruc_probs_train, self.struc_probs_train = generate_probs_2arm(
            self.probs, N=n_train_sessions, frac_impurity=frac_impurity
        )

        self.unstruc_probs_test, self.struc_probs_test = generate_probs_2arm(
            self.probs, N=n_test_sessions, frac_impurity=frac_impurity
        )

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.rng = np.random.default_rng()

        self.basedir = Path(basedir)
        self.root_folder = self.create_root_folder(
            n_train_sessions, n_test_sessions, frac_impurity
        )

    def create_root_folder(self, n_train_sessions, n_test_sessions, frac_impurity):
        folder_name = f"Train{n_train_sessions}Test{n_test_sessions}Impure{frac_impurity}_{self.timestamp}"
        root_folder = self.basedir / folder_name
        root_folder.mkdir(parents=True, exist_ok=True)
        return root_folder

    def _write_params_to_file(self, model_kwargs):
        params_df = pd.DataFrame(
            dict(
                params=[
                    "n_train_sessions",
                    "n_test_sessions",
                    "frac_impurity",
                    "hidden_size",
                    "lr",
                    "gamma",
                    "beta_entropy",
                    "beta_value",
                    "clip_norm",
                ],
                values=[
                    self.n_train_sessions,
                    self.n_test_sessions,
                    self.frac_impurity,
                    model_kwargs["hidden_size"],
                    model_kwargs["lr"],
                    model_kwargs["gamma"],
                    model_kwargs["beta_entropy"],
                    model_kwargs["beta_value"],
                    model_kwargs["clip_norm"],
                ],
            )
        )
        params_file = Path(self.root_folder / "params.csv")
        if not params_file.exists():
            params_df.to_csv(params_file, index=False)

    def _train_my_model(self, probs_train, model_path, model_kwargs):

        model = BanditTrainer2Arm(
            model_path=model_path,
            hidden_size=model_kwargs["hidden_size"],
            lr=model_kwargs["lr"],
            gamma=model_kwargs["gamma"],
            beta_entropy=model_kwargs["beta_entropy"],
            beta_value=model_kwargs["beta_value"],
            device=model_kwargs["device"],
        )
        model.train(
            n_sessions=self.n_train_sessions,
            mode=probs_train,
            clip_norm=model_kwargs["clip_norm"],
            progress_bar=False,
            save_model=False,
            return_df=False,
            n_trials=500,
        )
        return model

    def generate_model_data(
        self,
        name_suffix,
        train_type,
        hidden_size=48,
        lr=0.00004,
        gamma=0.92,
        beta_entropy=0.045,
        beta_value=0.025,
        clip_norm=4.5,
        performance_threshold=None,
    ):

        model_kwargs = {
            "hidden_size": hidden_size,
            "lr": lr,
            "gamma": gamma,
            "beta_entropy": beta_entropy,
            "beta_value": beta_value,
            "clip_norm": clip_norm,
            "device": "cpu",
        }
        self._write_params_to_file(model_kwargs)
        if train_type == "structured":
            probs_train = self.rng.permutation(self.struc_probs_train, axis=0)
            probs_test = self.rng.permutation(self.struc_probs_test, axis=0)

        if train_type == "unstructured":
            probs_train = self.rng.permutation(self.unstruc_probs_train, axis=0)
            probs_test = self.rng.permutation(self.unstruc_probs_test, axis=0)

        model_folder = self.root_folder / f"{train_type}_2arm_model{name_suffix}"
        model_folder.mkdir(exist_ok=True)
        model_path = model_folder / f"{train_type}_2arm_model{name_suffix}.pt"
        model_name = model_path.stem
        model = self._train_my_model(
            probs_train=probs_train, model_path=model_path, model_kwargs=model_kwargs
        )
        task_df, perf = self._evaluate_model(model, probs_test)

        if performance_threshold is None or perf >= performance_threshold:
            model.save_model()

            # ------ Congruent environment testing ------
            exp_folder = model_folder / f"{model_name}_{train_type}"
            exp_folder.mkdir(exist_ok=True)
            task_df.to_csv(exp_folder / f"{model_name}_{train_type}.csv", index=False)

            # ------ Incongruent environment testing ------

            other_train_type = (
                "unstructured" if train_type == "structured" else "structured"
            )
            other_exp_folder = model_folder / f"{model_name}_{other_train_type}"
            other_exp_folder.mkdir(exist_ok=True)
            other_probs_test = (
                self.rng.permutation(self.unstruc_probs_test, axis=0)
                if train_type == "structured"
                else self.rng.permutation(self.struc_probs_test, axis=0)
            )
            other_task_df = model.evaluate(
                mode=other_probs_test, n_sessions=self.n_test_sessions
            )
            other_task_df.to_csv(
                other_exp_folder / f"{model_name}_{other_train_type}.csv", index=False
            )

    def _evaluate_model(self, model, probs_test):
        task_df = model.evaluate(
            n_sessions=self.n_test_sessions,
            mode=probs_test,
            progress_bar=False,
        )

        task = Bandit2Arm.from_df(
            df=task_df,
            probs=["arm1_reward_prob", "arm2_reward_prob"],
            choices="chosen_action",
            rewards="reward",
            session_ids="session_id",
        )
        perf = task.get_optimal_choice_probability()[-5:].mean()
        return task_df, perf
