use std::cell::UnsafeCell;

use crate::Context;

#[derive(Debug, Clone)]
pub struct Expr<'a> {
    coefficent: &'a [isize],
    bias: isize,
}

impl<'a> Expr<'a> {
    pub fn new(ctx: &'a Context, coefficent: &[isize], bias: isize) -> Self {
        let coefficent = ctx.arena.alloc_slice_copy(coefficent);
        Self { coefficent, bias }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_expr_new<'a>(
    ctx: *mut Context,
    coefficent: *const isize,
    len: usize,
    bias: isize,
) -> *mut Expr<'a> {
    let ctx = &*ctx;
    let coefficent = unsafe { std::slice::from_raw_parts(coefficent, len) };
    ctx.arena
        .alloc(UnsafeCell::new(Expr::new(ctx, coefficent, bias)))
        .get_mut()
}
