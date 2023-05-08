
import torch
import enum



def normalize_quaternion(q):
    # Ensure the quaternion is a row vector
    q = q.view(1, -1) if q.dim() == 1 else q

    # Compute the norm of each quaternion
    norms = torch.norm(q, p=2, dim=1, keepdim=True)

    # Normalize each quaternion
    q = q / norms

    # Return the normalized quaternion
    return q.view(-1) if q.dim() == 2 and q.size(0) == 1 else q

class QuaternionCoeffOrder(enum.Enum):
    XYZW = 0
    WXYZ = 1


def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-8, order: QuaternionCoeffOrder = 0
) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                               order=QuaternionCoeffOrder.WXYZ)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

    # if order == QuaternionCoeffOrder.XYZW:
    #     warnings.warn(
    #         "`XYZW` quaternion coefficient order is deprecated and"
    #         " will be removed after > 0.6. "
    #         "Please use `QuaternionCoeffOrder.WXYZ` instead."
    #     )

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion

def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> torch.Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    # if not quaternion.shape[-1] == 4:
    #     raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # if not torch.jit.is_scripting():
    #     if order.name not in QuaternionCoeffOrder.__members__.keys():
    #         raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

    # if order == QuaternionCoeffOrder.XYZW:
    #     warnings.warn(
    #         "`XYZW` quaternion coefficient order is deprecated and"
    #         " will be removed after > 0.6. "
    #         "Please use `QuaternionCoeffOrder.WXYZ` instead."
    #     )

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    if order == QuaternionCoeffOrder.XYZW:
        x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)
    else:
        w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)

    matrix: torch.Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def tr(h): 

  """
  this function takes homogeneous trasformation matrices of dimension 1x4x4
  returns
  rotation of dimention       3x3
  translation of dimension    3x1  
  """
  h = h.squeeze()
  w_t_c =  h[:3, 3:4]
  c_R_w =  h[:3, :3].T.contiguous()


  return w_t_c , c_R_w

def reverse_tr(c_R_w, w_t_c):

    """
    this function takes a rotation matrix of dimension 3x3 and a translation matrix of dimension 3x1
    returns the original homogeneous transformation matrix of dimension 1x4x4
    """
#     print(c_R_w.shape, w_t_c.shape)
    h = torch.cat((c_R_w.squeeze(0).T, w_t_c.squeeze(0)), dim=1)
#     print(f"h = {h}")
    h = torch.cat((h, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0)


    return h.unsqueeze(0)
