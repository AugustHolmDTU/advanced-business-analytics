import unittest

from evch.train.sweep_mobile_rl import build_full_grid_variants


class MobileSweepTest(unittest.TestCase):
    def test_builds_full_grid_variants(self) -> None:
        base_config = {
            "experiment": {"name": "mobile_mcs_simple", "output_root": "outputs"},
            "environment": {"reward": {"active_mobile_station_cost": 45.0, "activation_cost": 12.0}},
            "logging": {"wandb": {"group": "local", "tags": ["dtu"]}},
            "sweep": {
                "name": "mobile_mcs_reward_sweep",
                "parameters": {
                    "environment.reward.active_mobile_station_cost": [35.0, 60.0],
                    "environment.reward.activation_cost": [8.0, 12.0],
                },
            },
        }

        variants = build_full_grid_variants(base_config, base_config["sweep"])

        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[0]["environment"]["reward"]["active_mobile_station_cost"], 35.0)
        self.assertEqual(variants[2]["environment"]["reward"]["activation_cost"], 8.0)
        self.assertEqual(variants[3]["environment"]["reward"]["activation_cost"], 12.0)
        self.assertEqual(variants[0]["logging"]["wandb"]["group"], "mobile_mcs_reward_sweep")
        self.assertIn("sweep", variants[0]["logging"]["wandb"]["tags"])
        self.assertEqual(variants[0]["sweep_run"]["mode"], "full_grid")
        self.assertEqual(variants[0]["sweep_run"]["combination_index"], 0)
        self.assertIn("environment.reward.active_mobile_station_cost", variants[0]["sweep_run"]["parameters"])


if __name__ == "__main__":
    unittest.main()
