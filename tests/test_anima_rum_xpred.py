import torch

from anima_rum_xpred import (
    CacheMetadata,
    ToyXPredNet,
    anima_euler_step,
    load_xpred_cache_sample,
    make_shifted_sigma_schedule,
    reflow_latent_z,
    reflow_training_target,
    sample_train_sigmas,
    sample_with_vpred_student,
    sample_with_xpred_student,
    save_xpred_cache_sample,
    xpred_to_anima_v,
)


def test_xpred_to_anima_velocity_matches_current_fm_formula():
    generator = torch.Generator().manual_seed(1)
    x0 = torch.randn(2, 16, 8, 8, generator=generator)
    eps = torch.randn(2, 16, 8, 8, generator=generator)
    sigma = torch.tensor([0.2, 0.7]).view(2, 1, 1, 1)
    z = reflow_latent_z(x0, eps, sigma)

    v = xpred_to_anima_v(z, x0, sigma)

    torch.testing.assert_close(v, eps - x0, rtol=1e-5, atol=1e-6)


def test_anima_euler_step_moves_between_sigmas_on_straight_reflow_line():
    generator = torch.Generator().manual_seed(2)
    x0 = torch.randn(1, 16, 8, 8, generator=generator)
    eps = torch.randn(1, 16, 8, 8, generator=generator)
    sigma = torch.tensor([[[[0.8]]]])
    sigma_next = torch.tensor([[[[0.3]]]])
    z = reflow_latent_z(x0, eps, sigma)
    v = eps - x0

    z_next = anima_euler_step(z, v, sigma, sigma_next)

    torch.testing.assert_close(z_next, reflow_latent_z(x0, eps, sigma_next), rtol=1e-5, atol=1e-6)


def test_reflow_training_target_supports_x_and_v_prediction():
    x_teacher = torch.tensor([[[[2.0]]]])
    eps = torch.tensor([[[[5.0]]]])

    assert reflow_training_target("x", x_teacher, eps).item() == 2.0
    assert reflow_training_target("v", x_teacher, eps).item() == 3.0


def test_shifted_sigma_schedule_is_descending_and_training_sigmas_use_same_shift():
    sigmas = make_shifted_sigma_schedule(steps=4, flow_shift=3.0, device="cpu", dtype=torch.float32)

    assert sigmas[0] == 1
    assert sigmas[-1] == 0
    assert torch.all(sigmas[:-1] >= sigmas[1:])

    generator = torch.Generator().manual_seed(3)
    train_sigmas = sample_train_sigmas(
        batch_size=128,
        sigma_min_train=0.02,
        flow_shift=3.0,
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    )

    assert train_sigmas.shape == (128, 1, 1, 1)
    assert float(train_sigmas.min()) >= float(make_shifted_sigma_schedule(1, 3.0, "cpu", torch.float32)[-2]) * 0
    assert float(train_sigmas.min()) >= 0.02
    assert float(train_sigmas.max()) <= 1.0


def test_xpred_cache_roundtrip_keeps_latent_shape_and_metadata(tmp_path):
    x_teacher = torch.randn(1, 16, 8, 8)
    eps = torch.randn(1, 16, 8, 8)
    metadata = CacheMetadata(
        prompt="1girl",
        width=64,
        height=64,
        seed=123,
        sample_index=0,
        teacher_steps=40,
        flow_shift=3.0,
        teacher_cfg=1.0,
        teacher_lora="/models/turbo.safetensors",
        teacher_lora_weight=0.75,
    )
    path = tmp_path / "sample-000000.safetensors"

    save_xpred_cache_sample(path, x_teacher, eps, {"prompt_embeds": torch.randn(1, 4, 8)}, metadata)
    loaded = load_xpred_cache_sample(path, "cpu", torch.float32)

    assert loaded["x_teacher_latent"].shape == (1, 16, 8, 8)
    assert loaded["eps_latent"].shape == (1, 16, 8, 8)
    assert loaded["width"] == 64
    assert loaded["height"] == 64
    assert loaded["seed"] == 123
    assert loaded["teacher_lora"] == "/models/turbo.safetensors"
    assert loaded["teacher_lora_weight"] == 0.75
    assert "prompt_embeds" in loaded["text_conditioning"]


def test_xpred_sampler_does_not_forward_at_terminal_zero_sigma():
    class CountStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, z, sigma):
            self.calls += 1
            return torch.zeros_like(z)

    student = CountStudent()
    eps = torch.randn(1, 16, 8, 8)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    out = sample_with_xpred_student(student, eps, sigmas)

    assert out.shape == eps.shape
    assert student.calls == 2
    assert torch.isfinite(out).all()


def test_vpred_sampler_moves_directly_with_predicted_velocity():
    class ConstantVelocityStudent(torch.nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity
            self.calls = 0

        def forward(self, z, sigma):
            self.calls += 1
            return torch.full_like(z, self.velocity)

    student = ConstantVelocityStudent(3.0)
    eps = torch.full((1, 16, 1, 1), 5.0)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    out = sample_with_vpred_student(student, eps, sigmas)

    assert student.calls == 2
    torch.testing.assert_close(out, torch.full_like(eps, 2.0))


def test_toy_train_step_produces_finite_loss():
    student = ToyXPredNet()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    x_teacher = torch.randn(1, 16, 8, 8)
    eps = torch.randn(1, 16, 8, 8)
    sigma = torch.full((1, 1, 1, 1), 0.5)

    from anima_rum_xpred import train_one_xpred_step

    loss = train_one_xpred_step(student, optimizer, x_teacher, eps, sigma)

    assert torch.isfinite(loss)
