def split_self_teammates(obs, self_dim, teammate_block_dim, n_teammates):
    s=obs[:self_dim]
    u_flat=obs[self_dim:self_dim+n_teammates*teammate_block_dim]
    u=[u_flat[i*teammate_block_dim:(i+1)*teammate_block_dim] for i in range(n_teammates)]
    return s,u

def permute_teammates(obs,self_dim,teammate_block_dim,n_teammates,perm):
    s,u=split_self_teammates(obs,self_dim,teammate_block_dim,n_teammates)
    up=[u[i] for i in perm]
    flat=[x for b in up for x in b]
    return s+flat
