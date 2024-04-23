import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
    Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(
        self,
        uaq: UniformAffineQuantizer,
        weight_tensor: torch.Tensor,
        round_mode="learned_round_sigmoid",
    ):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == "nearest":
            x_int = torch.round(x / self.delta)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(x / self.delta)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print("Draw stochastic sample")
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                # [My comment] If soft_targets == True, It find optimal V whan forward pass.
                x_int = x_floor + self.get_soft_targets()
            else:
                # [My comment] If soft_targets == False, It apply hard rounding.
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError("Wrong rounding mode")

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        """[My comment]
            h(V_(i,j)) = clip(sigmoid(V_(i,j)) * (zeta - gamma) + gamma), 0, 1)    ...(eq. 23)
        >>> return h(V_(i,j))
        """
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def init_alpha(self, x: torch.Tensor):
        """[My comment][main - layerRec - (1) init - init alpha]
        - 우리는 h(V)가 들어있는 loss를 최소화 해야함.
            w~ = s * clip (floor(w/s)+h(V_(i,j)),0,1)
        - 현재 산출된 rounding error를 h(V)라고 생각해야함.
        - 그런데 지금 최적화할 변수 V를 세팅해야함. V는 구해진적 없음.
        - 그래서 h()의 역함수인 -log(~~)를 취해, V를 구함. >> alpha에 대입.

        >>> alpha == V_(i,j) == learnable parameter
        >>> rest == h(V_(i,j)) == \delta W (rounding error)
        """
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == "learned_hard_sigmoid":
            print("Init alpha to be FP32")
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log(
                (self.zeta - self.gamma) / (rest - self.gamma) - 1
            )  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
