use std::{cell::UnsafeCell, ptr::NonNull};

use crate::{affine::Expr, Context};

#[derive(Debug, Clone)]
pub enum Graph<'a> {
    Start(Option<&'a Self>),
    End,
    Access {
        memref: usize,
        offset: &'a Expr<'a>,
        next: Option<&'a Self>,
    },
    Update {
        ivar: usize,
        expr: &'a Expr<'a>,
        next: Option<&'a Self>,
    },
    Branch {
        ivar: usize,
        bound: &'a Expr<'a>,
        then: Option<&'a Self>,
        r#else: Option<&'a Self>,
    },
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_new_start(
    ctx: *const Context,
    next: *mut Graph<'_>,
) -> *mut Graph<'_> {
    let ctx = &*ctx;
    let next = NonNull::new(next).map(|ptr| ptr.as_ref());
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Start(next)))
        .get_mut()
}

#[no_mangle]
pub extern "C" fn slpn_graph_new_end<'a>(ctx: *const Context) -> *mut Graph<'a> {
    let ctx = unsafe { &*ctx };
    ctx.arena.alloc(UnsafeCell::new(Graph::End)).get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_new_access<'a>(
    ctx: *const Context,
    memref: usize,
    offset: *mut Expr<'a>,
    next: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Access {
            memref,
            offset: &*offset,
            next: NonNull::new(next).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_new_update<'a>(
    ctx: *const Context,
    ivar: usize,
    expr: *mut Expr<'a>,
    next: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Update {
            ivar,
            expr: &*expr,
            next: NonNull::new(next).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_new_branch<'a>(
    ctx: *const Context,
    ivar: usize,
    bound: *mut Expr<'a>,
    then: *mut Graph<'a>,
    r#else: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Branch {
            ivar,
            bound: &*bound,
            then: NonNull::new(then).map(|ptr| ptr.as_ref()),
            r#else: NonNull::new(r#else).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[allow(improper_ctypes)]
extern "C" {
    fn slpn_extract_affine_loop<'a>(
        ctx: *const Context,
        filename: *const std::os::raw::c_char,
        length: usize,
    ) -> Option<NonNull<Graph<'a>>>;
}

impl<'a> Graph<'a> {
    pub fn new_from_file(ctx: &'a Context, filename: &str) -> Option<&'a Self> {
        let filename = std::ffi::CString::new(filename).unwrap();
        unsafe {
            let graph = slpn_extract_affine_loop(ctx, filename.as_ptr(), filename.as_bytes().len());
            graph.map(|graph| graph.as_ref())
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_start_set_next<'a>(
    start: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut start) = NonNull::new(start) {
        let start = start.as_mut();
        if let Graph::Start(ref mut field) = start {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        } else {
            panic!("expected start node");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_access_set_next<'a>(
    access: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut access) = NonNull::new(access) {
        let access = access.as_mut();
        if let Graph::Access {
            next: ref mut field,
            ..
        } = access
        {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        } else {
            panic!("expected access node");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_update_set_next<'a>(
    update: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut update) = NonNull::new(update) {
        let update = update.as_mut();
        if let Graph::Update {
            next: ref mut field,
            ..
        } = update
        {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        } else {
            panic!("expected update node");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_branch_set_then(branch: *mut Graph<'_>, then: *mut Graph) {
    if let Some(mut branch) = NonNull::new(branch) {
        let branch = branch.as_mut();
        if let Graph::Branch {
            then: ref mut field,
            ..
        } = branch
        {
            *field = NonNull::new(then).map(|ptr| ptr.as_ref());
        } else {
            panic!("expected branch node");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slpn_graph_branch_set_else(branch: *mut Graph<'_>, r#else: *mut Graph) {
    if let Some(mut branch) = NonNull::new(branch) {
        let branch = branch.as_mut();
        if let Graph::Branch {
            r#else: ref mut field,
            ..
        } = branch
        {
            *field = NonNull::new(r#else).map(|ptr| ptr.as_ref());
        } else {
            panic!("expected branch node");
        }
    }
}
